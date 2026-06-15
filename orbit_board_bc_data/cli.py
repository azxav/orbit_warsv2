from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .replay_loader import iter_replay_paths, load_replay
from .sample_builder import build_samples_from_replay
from .split import split_episode_ids
from .tensor_writer import (
    StreamingDatasetWriter,
    finalize_dataset_from_chunks,
    finalize_incremental_dataset_from_chunks,
    validate_existing_dataset_compatible,
    write_sample_chunk,
)
from .validator import validate_dataset


DEBUG_KEYS = [
    "unmatched_actions",
    "ambiguous_matches",
    "unknown_target_labels",
    "skipped_loser_turns",
    "extracted_action_target_labels",
]


@dataclass(frozen=True)
class ReplayBuildJob:
    path: str
    player_filter: str
    max_planets: int
    max_fleets: int
    max_actions: int
    noop_stop_weight: float
    keep_noop: bool
    target_hit_only: bool
    collect_debug: bool
    split: str = "train"
    chunk_dir: str | None = None
    result_id: str | None = None


@dataclass
class ReplayBuildResult:
    replay_stem: str
    samples: list
    debug: dict[str, list[dict[str, Any]]]
    debug_counts: dict[str, int]
    chunk_dir: str | None = None
    sample_count: int = 0


def _build_replay_path(job: ReplayBuildJob) -> ReplayBuildResult:
    path = Path(job.path)
    replay = load_replay(path)
    samples, replay_debug = build_samples_from_replay(
        replay,
        job.player_filter,
        job.max_planets,
        job.max_fleets,
        job.max_actions,
        job.noop_stop_weight,
        job.keep_noop,
        job.target_hit_only,
    )
    debug_counts = {key: len(replay_debug.get(key, [])) for key in DEBUG_KEYS}
    result_id = job.result_id or path.stem
    if job.chunk_dir is not None:
        if not samples:
            return ReplayBuildResult(result_id, [], replay_debug if job.collect_debug else {}, debug_counts, None, 0)
        write_sample_chunk(job.chunk_dir, samples)
        return ReplayBuildResult(result_id, [], replay_debug if job.collect_debug else {}, debug_counts, job.chunk_dir, len(samples))
    return ReplayBuildResult(result_id, samples, replay_debug if job.collect_debug else {}, debug_counts, None, len(samples))


def _iter_built_replays(jobs: list[ReplayBuildJob], workers: int):
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield _build_replay_path(job)
        return

    max_pending = max(1, workers * 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = set()
        next_job = 0
        while next_job < len(jobs) or pending:
            while next_job < len(jobs) and len(pending) < max_pending:
                pending.add(executor.submit(_build_replay_path, jobs[next_job]))
                next_job += 1
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count - 1, 8))


def _limit_replay_paths(replay_paths: list[Path], max_files: int | None) -> list[Path]:
    if max_files is None:
        return replay_paths
    if max_files < 1:
        raise SystemExit("--max-files must be greater than 0")
    return replay_paths[:max_files]


def _is_existing_dataset(path: str | Path) -> bool:
    root = Path(path)
    return (root / "dataset_info.json").exists() and (root / "train" / "sample_index.parquet").exists()


def _existing_episode_splits(dataset: str | Path) -> tuple[set[str], set[str]]:
    root = Path(dataset)
    train = set(pd.read_parquet(root / "train" / "sample_index.parquet")["episode_id"].astype(str))
    valid = set(pd.read_parquet(root / "valid" / "sample_index.parquet")["episode_id"].astype(str))
    return train, valid


def _append_valid_episode(episode_id: str, valid_ratio: float, seed: int) -> bool:
    if valid_ratio <= 0:
        return False
    if valid_ratio >= 1:
        return True
    digest = hashlib.blake2b(f"{seed}:{episode_id}".encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") / float(1 << 64)
    return value < valid_ratio


def _atomic_replace_dir(temp_dir: Path, target_dir: Path) -> None:
    backup_dir = target_dir.with_name(f"{target_dir.name}.__append_backup__")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    target_dir.rename(backup_dir)
    try:
        temp_dir.rename(target_dir)
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        backup_dir.rename(target_dir)
        raise
    shutil.rmtree(backup_dir)


def _cmd_build(args: argparse.Namespace) -> None:
    replay_paths = iter_replay_paths(args.replay_dir)
    if not replay_paths:
        raise SystemExit(f"No replay JSON files found in {args.replay_dir}")
    append_existing = bool(args.append and _is_existing_dataset(args.out_dir))
    replay_episode_ids: dict[Path, str] = {}
    train_ids: set[str]
    valid_ids: set[str]
    target_out_dir = Path(args.out_dir)
    build_out_dir = target_out_dir
    if append_existing:
        try:
            validate_existing_dataset_compatible(args.out_dir, args.max_planets, args.max_fleets, args.max_actions_per_turn)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        existing_train_ids, existing_valid_ids = _existing_episode_splits(args.out_dir)
        existing_ids = existing_train_ids | existing_valid_ids
        new_paths: list[Path] = []
        for path in replay_paths:
            episode_id = load_replay(path).episode_id
            replay_episode_ids[path] = episode_id
            if episode_id not in existing_ids:
                new_paths.append(path)
        if not new_paths:
            print("No new replay episodes found; dataset unchanged.")
            return
        replay_paths = _limit_replay_paths(new_paths, args.max_files)
        train_ids = set()
        valid_ids = set()
        for path in replay_paths:
            episode_id = replay_episode_ids[path]
            if _append_valid_episode(episode_id, args.valid_ratio, args.seed):
                valid_ids.add(episode_id)
            else:
                train_ids.add(episode_id)
        build_out_dir = target_out_dir.with_name(f"{target_out_dir.name}.__append_tmp__")
        if build_out_dir.exists():
            shutil.rmtree(build_out_dir)
    else:
        replay_paths = _limit_replay_paths(replay_paths, args.max_files)
        train_ids, valid_ids = split_episode_ids([path.stem for path in replay_paths], args.valid_ratio, args.seed)
    workers = max(1, min(int(args.workers), len(replay_paths)))
    use_worker_shards = args.worker_output == "shard" and workers > 1
    if append_existing:
        use_worker_shards = True
    writer = None
    if not use_worker_shards:
        writer = StreamingDatasetWriter(
            build_out_dir,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            max_actions=args.max_actions_per_turn,
            chunk_size=args.writer_chunk_size,
            compress_masks=args.compress_masks,
        )
    total_samples = 0
    debug: dict[str, list[dict[str, Any]]] = {key: [] for key in DEBUG_KEYS}
    quality_counts: dict[str, int] = {key: 0 for key in DEBUG_KEYS}
    split_chunks: dict[str, list[str]] = {"train": [], "valid": []}
    jobs = [
        ReplayBuildJob(
            path=str(path),
            player_filter=args.player_filter,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            max_actions=args.max_actions_per_turn,
            noop_stop_weight=args.noop_stop_weight,
            keep_noop=args.keep_noop,
            target_hit_only=args.target_hit_only,
            collect_debug=args.write_debug,
            split="valid" if (replay_episode_ids.get(path, path.stem)) in valid_ids else "train",
            chunk_dir=str(build_out_dir / "_worker_chunks" / ("valid" if (replay_episode_ids.get(path, path.stem)) in valid_ids else "train") / path.stem)
            if use_worker_shards
            else None,
            result_id=replay_episode_ids.get(path),
        )
        for path in replay_paths
    ]
    for result in _iter_built_replays(jobs, workers):
        samples = result.samples
        total_samples += result.sample_count
        for key, count in result.debug_counts.items():
            quality_counts[key] = quality_counts.get(key, 0) + count
        split = "valid" if result.replay_stem in valid_ids else "train"
        if result.chunk_dir is not None:
            split_chunks[split].append(result.chunk_dir)
        elif writer is not None:
            writer.add_samples(split, samples)
        if args.write_debug:
            for key, rows in result.debug.items():
                debug.setdefault(key, []).extend(rows)
    if not total_samples:
        raise SystemExit("No samples were built; check replay schema and filters")
    metadata_args = {key: value for key, value in vars(args).items() if key != "func"}
    metadata_args["workers"] = workers
    metadata_args["worker_output"] = "shard" if use_worker_shards else "parent"
    if append_existing:
        try:
            finalize_incremental_dataset_from_chunks(
                target_out_dir,
                build_out_dir,
                split_chunks,
                max_planets=args.max_planets,
                max_fleets=args.max_fleets,
                max_actions=args.max_actions_per_turn,
                debug=debug,
                args=metadata_args,
                quality_counts=quality_counts,
                compress_masks=args.compress_masks,
            )
            validate_dataset(build_out_dir)
            _atomic_replace_dir(build_out_dir, target_out_dir)
        except Exception:
            shutil.rmtree(build_out_dir, ignore_errors=True)
            raise
    elif use_worker_shards:
        finalize_dataset_from_chunks(
            build_out_dir,
            split_chunks,
            max_planets=args.max_planets,
            max_fleets=args.max_fleets,
            max_actions=args.max_actions_per_turn,
            debug=debug,
            args=metadata_args,
            quality_counts=quality_counts,
            compress_masks=args.compress_masks,
        )
    elif writer is not None:
        writer.finalize(debug, metadata_args, quality_counts=quality_counts)


def _cmd_validate(args: argparse.Namespace) -> None:
    report = validate_dataset(args.dataset, args.unmatched_threshold, args.ambiguous_threshold)
    print(json.dumps(report, indent=2))


def _cmd_feature_probe(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_globals = np.load(dataset / "train" / "global_features.npy")
    report = {
        "global_mean": train_globals.mean(axis=0).tolist() if len(train_globals) else [],
        "global_std": train_globals.std(axis=0).tolist() if len(train_globals) else [],
    }
    (out_dir / "feature_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m orbit_board_bc_data.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--replay-dir", required=True)
    build.add_argument("--out-dir", required=True)
    build.add_argument("--player-filter", choices=["winner", "top2", "all"], default="winner")
    build.add_argument("--valid-ratio", type=float, default=0.1)
    build.add_argument("--seed", type=int, default=13)
    build.add_argument("--max-files", type=int, default=None, help="Maximum number of replay JSON files to process during this build.")
    build.add_argument("--max-planets", type=int, default=40)
    build.add_argument("--max-fleets", type=int, default=256)
    build.add_argument("--max-actions-per-turn", type=int, default=32)
    build.add_argument("--keep-noop", action="store_true", default=False)
    build.add_argument("--noop-stop-weight", type=float, default=0.35)
    build.add_argument("--target-hit-only", action="store_true", default=False)
    build.add_argument("--write-debug", action="store_true", default=False)
    build.add_argument("--writer-chunk-size", type=int, default=2048)
    build.add_argument("--compress-masks", dest="compress_masks", action="store_true", default=True)
    build.add_argument("--no-compress-masks", dest="compress_masks", action="store_false")
    build.add_argument("--workers", type=int, default=_default_workers())
    build.add_argument("--worker-output", choices=["parent", "shard"], default="shard")
    build.add_argument("--append", action="store_true", default=False)
    build.set_defaults(func=_cmd_build)

    validate = sub.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--unmatched-threshold", type=float, default=0.01)
    validate.add_argument("--ambiguous-threshold", type=float, default=0.01)
    validate.set_defaults(func=_cmd_validate)

    probe = sub.add_parser("feature-probe")
    probe.add_argument("--dataset", required=True)
    probe.add_argument("--out-dir", required=True)
    probe.set_defaults(func=_cmd_feature_probe)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
