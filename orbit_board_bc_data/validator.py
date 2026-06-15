from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class ValidationError(RuntimeError):
    pass


def _load_split(path: Path) -> dict[str, np.ndarray]:
    arrays = {p.stem: np.load(p, allow_pickle=False) for p in path.glob("*.npy")}
    mask_fields = ["action_valid_mask", "source_candidate_mask", "target_candidate_mask"]
    if not all(field in arrays for field in mask_fields):
        masks = np.load(path / "action_masks.npz")
        arrays.update({k: masks[k] for k in masks.files})
    return arrays


def validate_dataset(dataset: str | Path, unmatched_threshold: float = 0.01, ambiguous_threshold: float = 0.01) -> dict:
    root = Path(dataset)
    report: dict = {"splits": {}}
    for split in ["train", "valid"]:
        split_path = root / split
        arrays = _load_split(split_path)
        n = arrays["planet_tokens"].shape[0]
        report["splits"][split] = {"samples": int(n)}
        if not np.isfinite(arrays["planet_tokens"]).all() or not np.isfinite(arrays["fleet_tokens"]).all():
            raise ValidationError(f"{split}: NaN/inf features")
        if arrays["global_features"].shape[0] != n:
            raise ValidationError(f"{split}: tensor dimension mismatch")
        if not arrays["action_stop_labels"].any(axis=1).all():
            raise ValidationError(f"{split}: missing STOP labels")
        if (arrays["action_ship_fraction_labels"] < 0).any() or (arrays["action_ship_fraction_labels"] > 1.0001).any():
            raise ValidationError(f"{split}: invalid ship_fraction")

    train_idx = pd.read_parquet(root / "train" / "sample_index.parquet")
    valid_idx = pd.read_parquet(root / "valid" / "sample_index.parquet")
    overlap = set(train_idx["episode_id"]).intersection(set(valid_idx["episode_id"]))
    if overlap:
        raise ValidationError(f"valid split contains train episode IDs: {sorted(overlap)[:5]}")

    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    stats = info.get("stats", {})
    matched = max(1, int(stats.get("matched_actions", 0)))
    unmatched_rate = int(stats.get("unmatched_actions", 0)) / matched
    ambiguous_rate = int(stats.get("ambiguous_matches", 0)) / matched
    report["quality"] = {"unmatched_rate": unmatched_rate, "ambiguous_rate": ambiguous_rate}
    if unmatched_rate > unmatched_threshold:
        raise ValidationError(f"unmatched action rate {unmatched_rate:.3f} exceeds {unmatched_threshold:.3f}")
    if ambiguous_rate > ambiguous_threshold:
        raise ValidationError(f"ambiguous match rate {ambiguous_rate:.3f} exceeds {ambiguous_threshold:.3f}")
    return report
