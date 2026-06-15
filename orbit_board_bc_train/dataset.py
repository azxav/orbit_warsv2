from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


ARRAY_KEYS = [
    "planet_tokens",
    "fleet_tokens",
    "global_features",
    "planet_masks",
    "fleet_masks",
    "action_source_labels",
    "action_target_labels",
    "action_angle_labels",
    "action_angle_offset_labels",
    "action_ship_fraction_labels",
    "action_stop_labels",
    "action_loss_weights",
    "turn_type",
    "phase_id",
    "winner_id",
]
MASK_KEYS = ["action_valid_mask", "source_candidate_mask", "target_candidate_mask"]


class BoardBCDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: str = "train",
        *,
        mmap_mode: str | None = "r",
        include_metadata: bool = True,
        array_keys: list[str] | tuple[str, ...] | set[str] | None = None,
    ):
        self.root = Path(dataset_dir) / split
        requested = set(ARRAY_KEYS + MASK_KEYS) if array_keys is None else set(array_keys)
        unknown = requested - set(ARRAY_KEYS + MASK_KEYS)
        if unknown:
            raise ValueError(f"Unknown dataset array keys: {sorted(unknown)}")
        self.arrays = {}
        for key in ARRAY_KEYS:
            if key in requested:
                self.arrays[key] = np.load(self.root / f"{key}.npy", mmap_mode=mmap_mode, allow_pickle=False)
        requested_masks = [key for key in MASK_KEYS if key in requested]
        if requested_masks and all((self.root / f"{key}.npy").exists() for key in requested_masks):
            for key in requested_masks:
                self.arrays[key] = np.load(self.root / f"{key}.npy", mmap_mode=mmap_mode, allow_pickle=False)
        elif requested_masks:
            masks = np.load(self.root / "action_masks.npz", allow_pickle=False)
            for key in requested_masks:
                self.arrays[key] = masks[key]
            masks.close()
        self.index = (
            pd.read_parquet(self.root / "sample_index.parquet", columns=["episode_id", "sample_step"])
            if include_metadata
            else None
        )

    def __len__(self) -> int:
        return int(self.arrays["planet_tokens"].shape[0])

    def __getitem__(self, idx: int) -> dict:
        item = {key: value[idx] for key, value in self.arrays.items()}
        if self.index is not None:
            item["episode_id"] = self.index.iloc[idx]["episode_id"]
            item["sample_step"] = int(self.index.iloc[idx]["sample_step"])
        return item
