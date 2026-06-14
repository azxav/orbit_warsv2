from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, config: dict, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "metrics": metrics}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)

