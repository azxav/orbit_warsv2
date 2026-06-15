from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict,
    metrics: dict,
    optimizer: torch.optim.Optimizer | None = None,
    train_state: dict | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "config": config, "metrics": metrics}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if train_state is not None:
        payload["train_state"] = train_state
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
