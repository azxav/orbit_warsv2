from __future__ import annotations

from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .collate import collate_samples
from .dataset import BoardBCDataset
from .model.board_bc_model import BoardBCModel
from .train_loop import evaluate_model, resolve_device


def load_model_from_checkpoint(path, device):
    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt["config"]
    model = BoardBCModel(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate(args) -> dict[str, float]:
    device = resolve_device(args.device)
    ds = BoardBCDataset(args.dataset, "valid")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    model, _ = load_model_from_checkpoint(args.checkpoint, device)
    return evaluate_model(model, loader, device)

