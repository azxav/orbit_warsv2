from __future__ import annotations

import math

import torch


@torch.no_grad()
def compute_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    valid = batch["action_valid_mask"]
    non_stop = valid & (~batch["action_stop_labels"]) & (batch["action_target_labels"] >= 0)
    source_pred = outputs["source_logits"].argmax(dim=-1)
    source_acc = ((source_pred == batch["action_source_labels"]) & valid).sum().float() / valid.sum().clamp_min(1)
    stop_pred = source_pred == outputs["source_logits"].shape[-1] - 1
    stop_acc = ((stop_pred == batch["action_stop_labels"]) & valid).sum().float() / valid.sum().clamp_min(1)
    if non_stop.any():
        target_logits = outputs["target_logits"]
        target_pred = target_logits.argmax(dim=-1)
        target_acc = ((target_pred == batch["action_target_labels"]) & non_stop).sum().float() / non_stop.sum().clamp_min(1)
        top3 = target_logits.topk(min(3, target_logits.shape[-1]), dim=-1).indices
        target_top3 = ((top3 == batch["action_target_labels"].unsqueeze(-1)) & non_stop.unsqueeze(-1)).any(dim=-1).sum().float() / non_stop.sum().clamp_min(1)
        angle_diff = (outputs["angle_offset"] - batch["action_angle_offset_labels"] + math.pi) % (2 * math.pi) - math.pi
        angle_mae = angle_diff[non_stop].abs().mean() * 180.0 / math.pi
        ship_mae = (outputs["ship_fraction"][non_stop] - batch["action_ship_fraction_labels"][non_stop]).abs().mean()
    else:
        target_acc = torch.tensor(0.0)
        target_top3 = torch.tensor(0.0)
        angle_mae = torch.tensor(0.0)
        ship_mae = torch.tensor(0.0)
    return {
        "source_accuracy": float(source_acc.cpu()),
        "target_top1_accuracy": float(target_acc.cpu()),
        "target_top3_accuracy": float(target_top3.cpu()),
        "angle_mae_degrees": float(angle_mae.cpu()),
        "ship_fraction_mae": float(ship_mae.cpu()),
        "stop_accuracy": float(stop_acc.cpu()),
    }

