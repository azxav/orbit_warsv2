from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def wrapped_angle_huber(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.2) -> torch.Tensor:
    diff = (pred - target + math.pi) % (2.0 * math.pi) - math.pi
    return F.huber_loss(diff, torch.zeros_like(diff), reduction="none", delta=beta)


def compute_board_bc_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    weights = batch["action_loss_weights"]
    valid = batch["action_valid_mask"]
    source_loss_all = F.cross_entropy(
        outputs["source_logits"].reshape(-1, outputs["source_logits"].shape[-1]),
        batch["action_source_labels"].reshape(-1),
        reduction="none",
    ).reshape_as(weights)
    source_loss = _weighted_mean(source_loss_all, weights)

    action_mask = valid & (~batch["action_stop_labels"]) & (batch["action_target_labels"] >= 0)
    if action_mask.any():
        target_loss_all = F.cross_entropy(
            outputs["target_logits"].reshape(-1, outputs["target_logits"].shape[-1]),
            batch["action_target_labels"].clamp_min(0).reshape(-1),
            reduction="none",
        ).reshape_as(weights)
        target_loss = _weighted_mean(target_loss_all, weights * action_mask.float())
        angle_loss = _weighted_mean(
            wrapped_angle_huber(outputs["angle_offset"], batch["action_angle_offset_labels"]),
            weights * action_mask.float(),
        )
        ship_loss = _weighted_mean(
            F.huber_loss(outputs["ship_fraction"], batch["action_ship_fraction_labels"], reduction="none", delta=0.1),
            weights * action_mask.float(),
        )
    else:
        z = outputs["source_logits"].sum() * 0.0
        target_loss = z
        angle_loss = z
        ship_loss = z
    total = source_loss + target_loss + 0.5 * angle_loss + ship_loss
    parts = {
        "source_loss": float(source_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "angle_loss": float(angle_loss.detach().cpu()),
        "ship_loss": float(ship_loss.detach().cpu()),
        "loss": float(total.detach().cpu()),
    }
    return total, parts

