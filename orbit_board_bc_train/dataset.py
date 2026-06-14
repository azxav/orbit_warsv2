from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BoardBCDataset(Dataset):
    def __init__(self, dataset_dir: str | Path, split: str = "train"):
        self.root = Path(dataset_dir) / split
        self.arrays = {
            "planet_tokens": np.load(self.root / "planet_tokens.npy"),
            "fleet_tokens": np.load(self.root / "fleet_tokens.npy"),
            "global_features": np.load(self.root / "global_features.npy"),
            "planet_masks": np.load(self.root / "planet_masks.npy"),
            "fleet_masks": np.load(self.root / "fleet_masks.npy"),
            "action_source_labels": np.load(self.root / "action_source_labels.npy"),
            "action_target_labels": np.load(self.root / "action_target_labels.npy"),
            "action_angle_labels": np.load(self.root / "action_angle_labels.npy"),
            "action_angle_offset_labels": np.load(self.root / "action_angle_offset_labels.npy"),
            "action_ship_fraction_labels": np.load(self.root / "action_ship_fraction_labels.npy"),
            "action_stop_labels": np.load(self.root / "action_stop_labels.npy"),
            "action_loss_weights": np.load(self.root / "action_loss_weights.npy"),
            "turn_type": np.load(self.root / "turn_type.npy"),
            "phase_id": np.load(self.root / "phase_id.npy"),
            "winner_id": np.load(self.root / "winner_id.npy"),
        }
        masks = np.load(self.root / "action_masks.npz")
        for key in masks.files:
            self.arrays[key] = masks[key]
        self.index = pd.read_parquet(self.root / "sample_index.parquet")

    def __len__(self) -> int:
        return int(self.arrays["planet_tokens"].shape[0])

    def __getitem__(self, idx: int) -> dict:
        item = {key: value[idx] for key, value in self.arrays.items()}
        item["episode_id"] = self.index.iloc[idx]["episode_id"]
        item["sample_step"] = int(self.index.iloc[idx]["sample_step"])
        return item

