from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Sized

import torch
from torch.utils.data import DataLoader, Sampler

from .checkpoint import load_checkpoint, save_checkpoint
from .collate import collate_samples
from .dataset import BoardBCDataset
from .losses import compute_board_bc_loss
from .metrics import compute_metrics
from .model.board_bc_model import BoardBCModel


TRAIN_ARRAY_KEYS = [
    "planet_tokens",
    "fleet_tokens",
    "global_features",
    "planet_masks",
    "fleet_masks",
    "action_source_labels",
    "action_target_labels",
    "action_angle_offset_labels",
    "action_ship_fraction_labels",
    "action_stop_labels",
    "action_loss_weights",
    "action_valid_mask",
    "source_candidate_mask",
    "target_candidate_mask",
]


class BlockShuffleSampler(Sampler[int]):
    def __init__(self, data_source: Sized, block_size: int = 65536) -> None:
        if block_size < 1:
            raise ValueError("block_size must be greater than 0")
        self.data_source = data_source
        self.block_size = int(block_size)

    def __iter__(self):
        sample_count = len(self.data_source)
        block_count = (sample_count + self.block_size - 1) // self.block_size
        for block_idx in torch.randperm(block_count).tolist():
            start = block_idx * self.block_size
            stop = min(start + self.block_size, sample_count)
            for offset in torch.randperm(stop - start).tolist():
                yield start + offset

    def __len__(self) -> int:
        return len(self.data_source)


class _TrainingLogger:
    def __init__(self, log_file: str | Path) -> None:
        self.path = Path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def info(self, message: str) -> None:
        print(message, flush=True)
        self._file.write(f"{message}\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "_TrainingLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _make_model(train_ds: BoardBCDataset, args) -> BoardBCModel:
    sample = train_ds[0]
    return BoardBCModel(
        planet_dim=sample["planet_tokens"].shape[-1],
        fleet_dim=sample["fleet_tokens"].shape[-1],
        global_dim=sample["global_features"].shape[-1],
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        heads=args.heads,
        dropout=args.dropout,
        max_actions=sample["action_source_labels"].shape[0],
        max_planets=sample["planet_tokens"].shape[0],
    )


def _checkpoint_config(train_ds: BoardBCDataset, args) -> dict:
    return {
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "planet_dim": train_ds[0]["planet_tokens"].shape[-1],
        "fleet_dim": train_ds[0]["fleet_tokens"].shape[-1],
        "global_dim": train_ds[0]["global_features"].shape[-1],
        "max_actions": train_ds[0]["action_source_labels"].shape[0],
        "max_planets": train_ds[0]["planet_tokens"].shape[0],
    }


def _validate_resume_config(checkpoint_config: dict, current_config: dict) -> None:
    mismatches = {
        key: (checkpoint_config.get(key), value)
        for key, value in current_config.items()
        if checkpoint_config.get(key) != value
    }
    if mismatches:
        details = ", ".join(f"{key}: checkpoint={old!r} current={new!r}" for key, (old, new) in mismatches.items())
        raise ValueError(f"Resume checkpoint is incompatible with current training config ({details})")


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            outputs = model(tensor_batch)
            loss, parts = compute_board_bc_loss(outputs, tensor_batch)
            metrics = compute_metrics(outputs, tensor_batch)
            merged = {**parts, **metrics, "eval_loss": float(loss.cpu())}
            bs = int(tensor_batch["planet_tokens"].shape[0])
            count += bs
            for key, value in merged.items():
                totals[key] = totals.get(key, 0.0) + float(value) * bs
    return {key: value / max(1, count) for key, value in totals.items()}


def _dataloader_kwargs(args, device: torch.device) -> dict:
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(getattr(args, "pin_memory", False) and device.type == "cuda"),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(getattr(args, "persistent_workers", False))
        prefetch_factor = getattr(args, "prefetch_factor", 2)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _resolve_log_file(args) -> Path:
    log_file = getattr(args, "log_file", None)
    if log_file:
        return Path(log_file)
    return Path(args.out_dir) / "train.log"


def _format_metrics(metrics: dict[str, float]) -> str:
    return json.dumps({key: round(float(value), 6) for key, value in metrics.items()}, sort_keys=True)


def train(args) -> dict[str, float]:
    device = resolve_device(args.device)
    train_ds = BoardBCDataset(args.dataset, "train", include_metadata=False, array_keys=TRAIN_ARRAY_KEYS)
    valid_ds = BoardBCDataset(args.dataset, "valid", include_metadata=False, array_keys=TRAIN_ARRAY_KEYS)
    loader_kwargs = _dataloader_kwargs(args, device)
    shuffle_block_size = int(getattr(args, "shuffle_block_size", 65536) or 0)
    train_sampler = BlockShuffleSampler(train_ds, shuffle_block_size) if shuffle_block_size > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_samples,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
        **loader_kwargs,
    )
    model = _make_model(train_ds, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_metrics: dict[str, float] = {}
    config = _checkpoint_config(train_ds, args)
    start_epoch = 1
    resume_path = getattr(args, "resume", None)
    log_interval = max(1, int(getattr(args, "log_interval", 10) or 1))
    with _TrainingLogger(_resolve_log_file(args)) as logger:
        logger.info(
            "Starting training "
            f"dataset={args.dataset} out_dir={args.out_dir} device={device} "
            f"train_samples={len(train_ds)} valid_samples={len(valid_ds)} "
            f"batch_size={args.batch_size} epochs={args.epochs} log_file={logger.path}"
        )
        if resume_path:
            ckpt = load_checkpoint(resume_path, map_location=device)
            _validate_resume_config(ckpt["config"], config)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" not in ckpt:
                raise ValueError("Resume checkpoint does not contain optimizer_state; use a training checkpoint saved with resume support")
            opt.load_state_dict(ckpt["optimizer_state"])
            train_state = ckpt.get("train_state", {})
            start_epoch = int(train_state.get("epoch", 0)) + 1
            best = float(train_state.get("best_eval_loss", ckpt.get("metrics", {}).get("eval_loss", best)))
            best_metrics = dict(train_state.get("best_metrics", ckpt.get("metrics", {})))
            logger.info(f"Resumed training from {resume_path} at epoch {start_epoch}/{args.epochs}")
            if start_epoch > args.epochs:
                logger.info(f"Training already complete at epoch {start_epoch - 1}/{args.epochs}")
                return best_metrics
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            epoch_start = time.perf_counter()
            batch_count = len(train_loader)
            running_loss = 0.0
            running_samples = 0
            for batch_idx, batch in enumerate(train_loader, start=1):
                tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                outputs = model(tensor_batch)
                loss, _ = compute_board_bc_loss(outputs, tensor_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                batch_samples = int(tensor_batch["planet_tokens"].shape[0])
                running_loss += float(loss.detach().cpu()) * batch_samples
                running_samples += batch_samples
                if batch_idx == 1 or batch_idx == batch_count or batch_idx % log_interval == 0:
                    avg_loss = running_loss / max(1, running_samples)
                    logger.info(
                        f"Epoch {epoch}/{args.epochs} batch {batch_idx}/{batch_count} "
                        f"train_loss={avg_loss:.6f}"
                    )
            metrics = evaluate_model(model, valid_loader, device)
            if metrics["eval_loss"] < best:
                best = metrics["eval_loss"]
                best_metrics = metrics
                save_checkpoint(
                    Path(args.out_dir) / "best.pt",
                    model,
                    config,
                    metrics,
                    opt,
                    {
                        "epoch": epoch,
                        "start_epoch": start_epoch,
                        "best_eval_loss": best,
                        "best_metrics": best_metrics,
                    },
                )
            save_checkpoint(
                Path(args.out_dir) / "last.pt",
                model,
                config,
                metrics,
                opt,
                {
                    "epoch": epoch,
                    "start_epoch": start_epoch,
                    "best_eval_loss": best,
                    "best_metrics": best_metrics,
                },
            )
            logger.info(
                f"Epoch {epoch}/{args.epochs} metrics {_format_metrics(metrics)} "
                f"elapsed={time.perf_counter() - epoch_start:.2f}s best_eval_loss={best:.6f}"
            )
    return best_metrics
