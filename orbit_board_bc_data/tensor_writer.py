from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .sample_builder import TurnSample
from .schema import DEFAULT_FEATURE_SCHEMA


ARRAY_FIELDS = [
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

MASK_FIELDS = [
    "action_valid_mask",
    "source_candidate_mask",
    "target_candidate_mask",
]


def _stack(samples: list[TurnSample], field: str) -> np.ndarray:
    values = [getattr(s, field) for s in samples]
    if field in {"turn_type", "phase_id", "winner_id"}:
        return np.asarray(values, dtype=np.int64)
    return np.stack(values, axis=0)


def _savez(path: Path, compress: bool, **arrays: np.ndarray) -> None:
    if compress:
        np.savez_compressed(path, **arrays)
        return
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for key, value in arrays.items():
            with archive.open(f"{key}.npy", mode="w", force_zip64=True) as fh:
                np.save(fh, value)


def write_split(out_dir: Path, samples: list[TurnSample]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in ARRAY_FIELDS:
        np.save(out_dir / f"{field}.npy", _stack(samples, field))
    np.savez_compressed(
        out_dir / "action_masks.npz",
        action_valid_mask=np.stack([s.action_valid_mask for s in samples]),
        source_candidate_mask=np.stack([s.source_candidate_mask for s in samples]),
        target_candidate_mask=np.stack([s.target_candidate_mask for s in samples]),
    )
    index = pd.DataFrame(
        {
            "episode_id": [s.episode_id for s in samples],
            "sample_step": [s.sample_step for s in samples],
            "winner_id": [s.winner_id for s in samples],
            "turn_type": [s.turn_type for s in samples],
            "phase_id": [s.phase_id for s in samples],
        }
    )
    index.to_parquet(out_dir / "sample_index.parquet", index=False)


def write_sample_chunk(chunk_dir: str | Path, samples: list[TurnSample]) -> None:
    path = Path(chunk_dir)
    path.mkdir(parents=True, exist_ok=True)
    for field in ARRAY_FIELDS:
        np.save(path / f"{field}.npy", _stack(samples, field))
    for field in MASK_FIELDS:
        np.save(path / f"{field}.npy", np.stack([getattr(s, field) for s in samples], axis=0))
    pd.DataFrame(
        {
            "episode_id": [s.episode_id for s in samples],
            "sample_step": [s.sample_step for s in samples],
            "winner_id": [s.winner_id for s in samples],
            "turn_type": [s.turn_type for s in samples],
            "phase_id": [s.phase_id for s in samples],
        }
    ).to_parquet(path / "sample_index.parquet", index=False)


def _empty_shape(field: str, max_planets: int, max_fleets: int, max_actions: int) -> tuple[int, ...]:
    planet_dim = len(DEFAULT_FEATURE_SCHEMA.planet_features)
    fleet_dim = len(DEFAULT_FEATURE_SCHEMA.fleet_features)
    global_dim = len(DEFAULT_FEATURE_SCHEMA.global_features)
    shapes = {
        "planet_tokens": (0, max_planets, planet_dim),
        "fleet_tokens": (0, max_fleets, fleet_dim),
        "global_features": (0, global_dim),
        "planet_masks": (0, max_planets),
        "fleet_masks": (0, max_fleets),
        "action_source_labels": (0, max_actions),
        "action_target_labels": (0, max_actions),
        "action_angle_labels": (0, max_actions),
        "action_angle_offset_labels": (0, max_actions),
        "action_ship_fraction_labels": (0, max_actions),
        "action_stop_labels": (0, max_actions),
        "action_loss_weights": (0, max_actions),
        "action_valid_mask": (0, max_actions),
        "source_candidate_mask": (0, max_actions, max_planets + 1),
        "target_candidate_mask": (0, max_actions, max_planets),
        "turn_type": (0,),
        "phase_id": (0,),
        "winner_id": (0,),
    }
    return shapes[field]


def _dtype_for_field(field: str) -> Any:
    if field in {"planet_tokens", "fleet_tokens", "global_features", "action_angle_labels", "action_angle_offset_labels", "action_ship_fraction_labels", "action_loss_weights"}:
        return np.float32
    if field in {"planet_masks", "fleet_masks", "action_stop_labels"}:
        return bool
    return np.int64


def _write_empty_index(path: Path) -> None:
    pd.DataFrame(
        {
            "episode_id": pd.Series(dtype="object"),
            "sample_step": pd.Series(dtype="int64"),
            "winner_id": pd.Series(dtype="int64"),
            "turn_type": pd.Series(dtype="int64"),
            "phase_id": pd.Series(dtype="int64"),
        }
    ).to_parquet(path, index=False)


def _merge_chunk_field(chunks: list[Path], field: str, out_path: Path, sample_count: int, empty_shape: tuple[int, ...]) -> None:
    if not chunks:
        np.save(out_path, np.zeros(empty_shape, dtype=_dtype_for_field(field)))
        return
    first = np.load(chunks[0] / f"{field}.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=first.dtype, shape=(sample_count,) + first.shape[1:])
    offset = 0
    for chunk in chunks:
        arr = np.load(chunk / f"{field}.npy", mmap_mode="r")
        next_offset = offset + arr.shape[0]
        out[offset:next_offset] = arr
        offset = next_offset
    out.flush()


def _finalize_split_from_chunks(
    split_dir: Path,
    chunks: list[Path],
    max_planets: int,
    max_fleets: int,
    max_actions: int,
    compress_masks: bool,
) -> dict[str, int]:
    split_dir.mkdir(parents=True, exist_ok=True)
    sample_count = 0
    action_turns = 0
    noop_turns = 0
    for chunk in chunks:
        turn_type = np.load(chunk / "turn_type.npy", mmap_mode="r")
        sample_count += int(turn_type.shape[0])
        action_turns += int((turn_type == 1).sum())
        noop_turns += int((turn_type == 0).sum())

    for field in ARRAY_FIELDS:
        _merge_chunk_field(chunks, field, split_dir / f"{field}.npy", sample_count, _empty_shape(field, max_planets, max_fleets, max_actions))

    mask_paths: dict[str, Path] = {}
    for field in MASK_FIELDS:
        path = split_dir / f"_{field}.npy"
        _merge_chunk_field(chunks, field, path, sample_count, _empty_shape(field, max_planets, max_fleets, max_actions))
        mask_paths[field] = path
    _savez(
        split_dir / "action_masks.npz",
        compress_masks,
        action_valid_mask=np.load(mask_paths["action_valid_mask"], mmap_mode="r"),
        source_candidate_mask=np.load(mask_paths["source_candidate_mask"], mmap_mode="r"),
        target_candidate_mask=np.load(mask_paths["target_candidate_mask"], mmap_mode="r"),
    )
    for path in mask_paths.values():
        path.unlink(missing_ok=True)

    index_writer: pq.ParquetWriter | None = None
    index_path = split_dir / "sample_index.parquet"
    for chunk in chunks:
        table = pq.read_table(chunk / "sample_index.parquet")
        if index_writer is None:
            index_writer = pq.ParquetWriter(index_path, table.schema)
        index_writer.write_table(table)
    if index_writer is not None:
        index_writer.close()
    else:
        _write_empty_index(index_path)
    return {"samples": sample_count, "action_turns": action_turns, "noop_turns": noop_turns}


def finalize_dataset_from_chunks(
    out_dir: str | Path,
    split_chunks: dict[str, list[str | Path]],
    max_planets: int,
    max_fleets: int,
    max_actions: int,
    debug: dict[str, list[dict[str, Any]]],
    args: dict[str, Any],
    quality_counts: dict[str, int] | None = None,
    compress_masks: bool = True,
) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_stats = _finalize_split_from_chunks(
        root / "train",
        [Path(path) for path in split_chunks.get("train", [])],
        max_planets,
        max_fleets,
        max_actions,
        compress_masks,
    )
    valid_stats = _finalize_split_from_chunks(
        root / "valid",
        [Path(path) for path in split_chunks.get("valid", [])],
        max_planets,
        max_fleets,
        max_actions,
        compress_masks,
    )
    (root / "debug").mkdir(exist_ok=True)
    for name, rows in debug.items():
        pd.DataFrame(rows).to_csv(root / "debug" / f"{name}.csv", index=False)
    counts = quality_counts or {key: len(rows) for key, rows in debug.items()}
    stats = {
        "train_samples": train_stats["samples"],
        "valid_samples": valid_stats["samples"],
        "action_turns": train_stats["action_turns"] + valid_stats["action_turns"],
        "noop_turns": train_stats["noop_turns"] + valid_stats["noop_turns"],
        "unmatched_actions": int(counts.get("unmatched_actions", 0)),
        "ambiguous_matches": int(counts.get("ambiguous_matches", 0)),
        "unknown_target_labels": int(counts.get("unknown_target_labels", 0)),
        "matched_actions": int(counts.get("extracted_action_target_labels", 0)),
    }
    (root / "dataset_info.json").write_text(json.dumps({"args": args, "stats": stats}, indent=2), encoding="utf-8")
    (root / "feature_schema.json").write_text(json.dumps(asdict(DEFAULT_FEATURE_SCHEMA), indent=2), encoding="utf-8")
    (root / "label_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    shutil.rmtree(root / "_worker_chunks", ignore_errors=True)


class _SplitChunkWriter:
    def __init__(
        self,
        root: Path,
        split: str,
        max_planets: int,
        max_fleets: int,
        max_actions: int,
        chunk_size: int,
        compress_masks: bool,
    ) -> None:
        self.split_dir = root / split
        self.chunk_dir = root / "_chunks" / split
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.max_actions = max_actions
        self.chunk_size = chunk_size
        self.compress_masks = compress_masks
        self.buffer: list[TurnSample] = []
        self.chunks: list[Path] = []
        self.sample_count = 0
        self.action_turns = 0
        self.noop_turns = 0
        self._index_writer: pq.ParquetWriter | None = None
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

    def add_samples(self, samples: list[TurnSample]) -> None:
        for sample in samples:
            self.buffer.append(sample)
            self.sample_count += 1
            self.action_turns += int(sample.turn_type == 1)
            self.noop_turns += int(sample.turn_type == 0)
            if len(self.buffer) >= self.chunk_size:
                self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        chunk = self.chunk_dir / f"chunk-{len(self.chunks):06d}"
        chunk.mkdir(parents=True, exist_ok=True)
        for field in ARRAY_FIELDS:
            np.save(chunk / f"{field}.npy", _stack(self.buffer, field))
        for field in MASK_FIELDS:
            np.save(chunk / f"{field}.npy", np.stack([getattr(s, field) for s in self.buffer], axis=0))
        self._write_index_rows(self.buffer)
        self.chunks.append(chunk)
        self.buffer = []

    def _write_index_rows(self, samples: list[TurnSample]) -> None:
        table = pa.Table.from_pydict(
            {
                "episode_id": [s.episode_id for s in samples],
                "sample_step": [s.sample_step for s in samples],
                "winner_id": [s.winner_id for s in samples],
                "turn_type": [s.turn_type for s in samples],
                "phase_id": [s.phase_id for s in samples],
            }
        )
        if self._index_writer is None:
            self._index_writer = pq.ParquetWriter(self.split_dir / "sample_index.parquet", table.schema)
        self._index_writer.write_table(table)

    def finalize(self) -> None:
        self.flush()
        if self._index_writer is not None:
            self._index_writer.close()
        else:
            self._write_empty_index()
        for field in ARRAY_FIELDS:
            self._merge_field(field, self.split_dir / f"{field}.npy", self._empty_shape(field), self._dtype_for_field(field))

        mask_paths: dict[str, Path] = {}
        for field in MASK_FIELDS:
            path = self.split_dir / f"_{field}.npy"
            self._merge_field(field, path, self._empty_shape(field), bool)
            mask_paths[field] = path
        _savez(
            self.split_dir / "action_masks.npz",
            self.compress_masks,
            action_valid_mask=np.load(mask_paths["action_valid_mask"], mmap_mode="r"),
            source_candidate_mask=np.load(mask_paths["source_candidate_mask"], mmap_mode="r"),
            target_candidate_mask=np.load(mask_paths["target_candidate_mask"], mmap_mode="r"),
        )
        for path in mask_paths.values():
            path.unlink(missing_ok=True)

    def _write_empty_index(self) -> None:
        pd.DataFrame(
            {
                "episode_id": pd.Series(dtype="object"),
                "sample_step": pd.Series(dtype="int64"),
                "winner_id": pd.Series(dtype="int64"),
                "turn_type": pd.Series(dtype="int64"),
                "phase_id": pd.Series(dtype="int64"),
            }
        ).to_parquet(self.split_dir / "sample_index.parquet", index=False)

    def _merge_field(self, field: str, out_path: Path, empty_shape: tuple[int, ...], empty_dtype: Any) -> None:
        if not self.chunks:
            np.save(out_path, np.zeros(empty_shape, dtype=empty_dtype))
            return
        first = np.load(self.chunks[0] / f"{field}.npy", mmap_mode="r")
        shape = (self.sample_count,) + first.shape[1:]
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=first.dtype, shape=shape)
        offset = 0
        for chunk in self.chunks:
            arr = np.load(chunk / f"{field}.npy", mmap_mode="r")
            next_offset = offset + arr.shape[0]
            out[offset:next_offset] = arr
            offset = next_offset
        out.flush()

    def _empty_shape(self, field: str) -> tuple[int, ...]:
        planet_dim = len(DEFAULT_FEATURE_SCHEMA.planet_features)
        fleet_dim = len(DEFAULT_FEATURE_SCHEMA.fleet_features)
        global_dim = len(DEFAULT_FEATURE_SCHEMA.global_features)
        shapes = {
            "planet_tokens": (0, self.max_planets, planet_dim),
            "fleet_tokens": (0, self.max_fleets, fleet_dim),
            "global_features": (0, global_dim),
            "planet_masks": (0, self.max_planets),
            "fleet_masks": (0, self.max_fleets),
            "action_source_labels": (0, self.max_actions),
            "action_target_labels": (0, self.max_actions),
            "action_angle_labels": (0, self.max_actions),
            "action_angle_offset_labels": (0, self.max_actions),
            "action_ship_fraction_labels": (0, self.max_actions),
            "action_stop_labels": (0, self.max_actions),
            "action_loss_weights": (0, self.max_actions),
            "action_valid_mask": (0, self.max_actions),
            "source_candidate_mask": (0, self.max_actions, self.max_planets + 1),
            "target_candidate_mask": (0, self.max_actions, self.max_planets),
            "turn_type": (0,),
            "phase_id": (0,),
            "winner_id": (0,),
        }
        return shapes[field]

    @staticmethod
    def _dtype_for_field(field: str) -> Any:
        if field in {"planet_tokens", "fleet_tokens", "global_features", "action_angle_labels", "action_angle_offset_labels", "action_ship_fraction_labels", "action_loss_weights"}:
            return np.float32
        if field in {"planet_masks", "fleet_masks", "action_stop_labels"}:
            return bool
        return np.int64


class StreamingDatasetWriter:
    def __init__(
        self,
        out_dir: str | Path,
        max_planets: int,
        max_fleets: int,
        max_actions: int,
        chunk_size: int = 2048,
        compress_masks: bool = True,
    ) -> None:
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self._writers = {
            "train": _SplitChunkWriter(self.root, "train", max_planets, max_fleets, max_actions, chunk_size, compress_masks),
            "valid": _SplitChunkWriter(self.root, "valid", max_planets, max_fleets, max_actions, chunk_size, compress_masks),
        }

    def add_samples(self, split: str, samples: list[TurnSample]) -> None:
        self._writers[split].add_samples(samples)

    def finalize(
        self,
        debug: dict[str, list[dict[str, Any]]],
        args: dict[str, Any],
        quality_counts: dict[str, int] | None = None,
    ) -> None:
        for writer in self._writers.values():
            writer.finalize()
        (self.root / "debug").mkdir(exist_ok=True)
        for name, rows in debug.items():
            pd.DataFrame(rows).to_csv(self.root / "debug" / f"{name}.csv", index=False)
        counts = quality_counts or {key: len(rows) for key, rows in debug.items()}
        stats = {
            "train_samples": self._writers["train"].sample_count,
            "valid_samples": self._writers["valid"].sample_count,
            "action_turns": self._writers["train"].action_turns + self._writers["valid"].action_turns,
            "noop_turns": self._writers["train"].noop_turns + self._writers["valid"].noop_turns,
            "unmatched_actions": int(counts.get("unmatched_actions", 0)),
            "ambiguous_matches": int(counts.get("ambiguous_matches", 0)),
            "unknown_target_labels": int(counts.get("unknown_target_labels", 0)),
            "matched_actions": int(counts.get("extracted_action_target_labels", 0)),
        }
        (self.root / "dataset_info.json").write_text(json.dumps({"args": args, "stats": stats}, indent=2), encoding="utf-8")
        (self.root / "feature_schema.json").write_text(json.dumps(asdict(DEFAULT_FEATURE_SCHEMA), indent=2), encoding="utf-8")
        (self.root / "label_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        shutil.rmtree(self.root / "_chunks", ignore_errors=True)


def write_dataset(
    out_dir: str | Path,
    train_samples: list[TurnSample],
    valid_samples: list[TurnSample],
    debug: dict[str, list[dict[str, Any]]],
    args: dict[str, Any],
) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_split(root / "train", train_samples)
    write_split(root / "valid", valid_samples if valid_samples else train_samples[:0])
    (root / "debug").mkdir(exist_ok=True)
    for name, rows in debug.items():
        pd.DataFrame(rows).to_csv(root / "debug" / f"{name}.csv", index=False)
    stats = {
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "action_turns": int(sum(s.turn_type == 1 for s in train_samples + valid_samples)),
        "noop_turns": int(sum(s.turn_type == 0 for s in train_samples + valid_samples)),
        "unmatched_actions": len(debug.get("unmatched_actions", [])),
        "ambiguous_matches": len(debug.get("ambiguous_matches", [])),
        "unknown_target_labels": len(debug.get("unknown_target_labels", [])),
        "matched_actions": len(debug.get("extracted_action_target_labels", [])),
    }
    (root / "dataset_info.json").write_text(json.dumps({"args": args, "stats": stats}, indent=2), encoding="utf-8")
    (root / "feature_schema.json").write_text(json.dumps(asdict(DEFAULT_FEATURE_SCHEMA), indent=2), encoding="utf-8")
    (root / "label_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
