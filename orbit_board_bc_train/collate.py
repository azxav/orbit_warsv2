from __future__ import annotations

import numpy as np
import torch


FLOAT_KEYS = {
    "planet_tokens",
    "fleet_tokens",
    "global_features",
    "action_angle_labels",
    "action_angle_offset_labels",
    "action_ship_fraction_labels",
    "action_loss_weights",
}
BOOL_KEYS = {
    "planet_masks",
    "fleet_masks",
    "action_stop_labels",
    "action_valid_mask",
    "source_candidate_mask",
    "target_candidate_mask",
}
LONG_KEYS = {"action_source_labels", "action_target_labels", "turn_type", "phase_id", "winner_id"}


def collate_samples(samples: list[dict]) -> dict[str, torch.Tensor | list]:
    batch: dict[str, torch.Tensor | list] = {}
    keys = [k for k in samples[0] if k not in {"episode_id", "sample_step"}]
    for key in keys:
        arr = np.stack([s[key] for s in samples], axis=0)
        if key in FLOAT_KEYS:
            batch[key] = torch.as_tensor(arr, dtype=torch.float32)
        elif key in BOOL_KEYS:
            batch[key] = torch.as_tensor(arr, dtype=torch.bool)
        elif key in LONG_KEYS:
            batch[key] = torch.as_tensor(arr, dtype=torch.long)
        else:
            batch[key] = torch.as_tensor(arr)
    if "episode_id" in samples[0]:
        batch["episode_id"] = [s["episode_id"] for s in samples]
    if "sample_step" in samples[0]:
        batch["sample_step"] = torch.as_tensor([s["sample_step"] for s in samples], dtype=torch.long)
    return batch
