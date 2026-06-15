from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .collate import collate_samples
from .dataset import BoardBCDataset
from .losses import compute_board_bc_loss
from .metrics import compute_metrics
from .model.board_bc_model import BoardBCModel


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


def train(args) -> dict[str, float]:
    device = resolve_device(args.device)
    train_ds = BoardBCDataset(args.dataset, "train")
    valid_ds = BoardBCDataset(args.dataset, "valid")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_samples)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    model = _make_model(train_ds, args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    best_metrics: dict[str, float] = {}
    config = _checkpoint_config(train_ds, args)
    start_epoch = 1
    resume_path = getattr(args, "resume", None)
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
        if start_epoch > args.epochs:
            return best_metrics
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for batch in train_loader:
            tensor_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            outputs = model(tensor_batch)
            loss, _ = compute_board_bc_loss(outputs, tensor_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
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
        print({"epoch": epoch, **metrics})
    return best_metrics
