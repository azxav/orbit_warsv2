from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .feature_builder import BoardFeatures, build_board_features
from .fleet_matching import match_created_fleet
from .geometry import angle_between, wrap_angle
from .replay_index import IndexedReplayFrames
from .replay_loader import Replay, find_winner_id
from .split import phase_id, select_players
from .target_extraction import TARGET_UNKNOWN, extract_target_for_fleet

PLANET_ID = 0
PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_SHIPS = 5


@dataclass
class TurnSample:
    planet_tokens: np.ndarray
    fleet_tokens: np.ndarray
    global_features: np.ndarray
    planet_masks: np.ndarray
    fleet_masks: np.ndarray
    action_source_labels: np.ndarray
    action_target_labels: np.ndarray
    action_angle_labels: np.ndarray
    action_angle_offset_labels: np.ndarray
    action_ship_fraction_labels: np.ndarray
    action_stop_labels: np.ndarray
    action_loss_weights: np.ndarray
    action_valid_mask: np.ndarray
    source_candidate_mask: np.ndarray
    target_candidate_mask: np.ndarray
    turn_type: int
    phase_id: int
    winner_id: int
    episode_id: str
    sample_step: int
    planet_ids: list[int]


def _planet_index(features: BoardFeatures, planet_id: int) -> int | None:
    try:
        return features.planet_ids.index(int(planet_id))
    except ValueError:
        return None


def _source_candidate_mask(
    player_id: int,
    features: BoardFeatures,
    planet_by_id: dict[int, list[float]],
    max_planets: int,
    max_actions: int,
) -> np.ndarray:
    mask = np.zeros((max_actions, max_planets + 1), dtype=bool)
    for i, pid in enumerate(features.planet_ids[:max_planets]):
        p = planet_by_id.get(pid)
        if p is not None and int(p[PLANET_OWNER]) == player_id and float(p[PLANET_SHIPS]) > 0:
            mask[:, i] = True
    mask[:, max_planets] = True
    return mask


def _target_candidate_mask(features: BoardFeatures, max_planets: int, max_actions: int) -> np.ndarray:
    mask = np.zeros((max_actions, max_planets), dtype=bool)
    mask[:, : len(features.planet_ids)] = True
    return mask


def build_turn_sample(
    obs: dict,
    player_id: int,
    episode_id: str,
    sample_step: int,
    extracted_actions: list[dict[str, Any]],
    max_planets: int,
    max_fleets: int,
    max_actions: int,
    noop_stop_weight: float,
    winner_id: int | None = None,
) -> TurnSample | None:
    features = build_board_features(obs, player_id, max_planets, max_fleets)
    stop_idx = max_planets
    source_labels = np.full((max_actions,), stop_idx, dtype=np.int64)
    target_labels = np.full((max_actions,), -100, dtype=np.int64)
    angle_labels = np.zeros((max_actions,), dtype=np.float32)
    angle_offset_labels = np.zeros((max_actions,), dtype=np.float32)
    ship_fraction_labels = np.zeros((max_actions,), dtype=np.float32)
    stop_labels = np.ones((max_actions,), dtype=bool)
    loss_weights = np.zeros((max_actions,), dtype=np.float32)
    valid_mask = np.zeros((max_actions,), dtype=bool)
    planet_by_id = {int(p[PLANET_ID]): p for p in obs.get("planets", [])}
    planet_index_by_id = {pid: idx for idx, pid in enumerate(features.planet_ids)}

    limit = min(len(extracted_actions), max_actions - 1)
    for j, action in enumerate(extracted_actions[:limit]):
        source_idx = planet_index_by_id.get(int(action["source_id"]))
        target_idx = planet_index_by_id.get(int(action["target_id"]))
        source_planet = planet_by_id.get(int(action["source_id"]))
        target_planet = planet_by_id.get(int(action["target_id"]))
        if source_idx is None or target_idx is None or source_planet is None or target_planet is None:
            return None
        source_owner = int(source_planet[PLANET_OWNER])
        source_ships = float(source_planet[PLANET_SHIPS])
        if source_owner != player_id or source_ships <= 0:
            return None
        frac = float(action["num_ships"]) / max(1.0, source_ships)
        source_labels[j] = source_idx
        target_labels[j] = target_idx
        angle = float(action["angle"])
        angle_labels[j] = angle
        geom = angle_between(
            float(source_planet[PLANET_X]),
            float(source_planet[PLANET_Y]),
            float(target_planet[PLANET_X]),
            float(target_planet[PLANET_Y]),
        )
        angle_offset_labels[j] = wrap_angle(angle - geom)
        ship_fraction_labels[j] = frac
        stop_labels[j] = False
        loss_weights[j] = 1.0
        valid_mask[j] = True

    stop_pos = limit
    source_labels[stop_pos] = stop_idx
    stop_labels[stop_pos] = True
    loss_weights[stop_pos] = noop_stop_weight if limit == 0 else 1.0
    valid_mask[stop_pos] = True

    return TurnSample(
        features.planet_tokens,
        features.fleet_tokens,
        features.global_features,
        features.planet_mask,
        features.fleet_mask,
        source_labels,
        target_labels,
        angle_labels,
        angle_offset_labels,
        ship_fraction_labels,
        stop_labels,
        loss_weights,
        valid_mask,
        _source_candidate_mask(player_id, features, planet_by_id, max_planets, max_actions),
        _target_candidate_mask(features, max_planets, max_actions),
        0 if limit == 0 else 1,
        phase_id(sample_step),
        int(winner_id if winner_id is not None else player_id),
        episode_id,
        sample_step,
        features.planet_ids,
    )


def extract_actions_for_turn(
    replay: Replay,
    player_id: int,
    row_index: int,
    target_hit_only: bool,
    indexed_frames: IndexedReplayFrames | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    before = replay.steps[row_index - 1][player_id]["observation"]
    after = replay.steps[row_index][player_id]["observation"]
    actions = replay.steps[row_index][player_id].get("action") or []
    frames = indexed_frames if indexed_frames is not None else [replay.steps[i][player_id]["observation"] for i in range(row_index, len(replay.steps))]
    target_start_index = row_index if indexed_frames is not None else 0
    extracted: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    unknown_targets: list[dict[str, Any]] = []
    ship_speed = float(replay.configuration.get("shipSpeed", 6.0))
    before_fleet_ids = indexed_frames.fleet_ids(row_index - 1) if indexed_frames is not None else None
    after_fleet_rows = indexed_frames.fleet_rows(row_index) if indexed_frames is not None else None
    for action in actions:
        match = match_created_fleet(
            action,
            before,
            after,
            player_id,
            before_fleet_ids=before_fleet_ids,
            after_fleet_rows=after_fleet_rows,
        )
        if match is None:
            unmatched.append({"episode_id": replay.episode_id, "step": row_index, "player_id": player_id, "action": action})
            continue
        if match.status == "ambiguous":
            ambiguous.append({"episode_id": replay.episode_id, "step": row_index, "player_id": player_id, "fleet_id": match.fleet_id})
            continue
        target = extract_target_for_fleet(match.fleet_id, frames, target_start_index, ship_speed)
        if target.target_id < 0 and target_hit_only:
            unknown_targets.append(
                {
                    "episode_id": replay.episode_id,
                    "step": row_index,
                    "player_id": player_id,
                    "fleet_id": match.fleet_id,
                    "target_status": target.status,
                }
            )
            continue
        if target.target_id == TARGET_UNKNOWN:
            continue
        extracted.append(
            {
                "source_id": int(action[0]),
                "target_id": int(target.target_id),
                "angle": float(action[1]),
                "num_ships": float(action[2]),
                "fleet_id": int(match.fleet_id),
                "target_status": target.status,
            }
        )
    return extracted, unmatched, ambiguous, unknown_targets


def build_samples_from_replay(
    replay: Replay,
    player_filter: str,
    max_planets: int,
    max_fleets: int,
    max_actions: int,
    noop_stop_weight: float,
    keep_noop: bool,
    target_hit_only: bool,
) -> tuple[list[TurnSample], dict[str, list[dict[str, Any]]]]:
    winner_id = find_winner_id(replay)
    selected = set(select_players(replay, player_filter))
    debug: dict[str, list[dict[str, Any]]] = {
        "unmatched_actions": [],
        "ambiguous_matches": [],
        "unknown_target_labels": [],
        "skipped_loser_turns": [],
        "extracted_action_target_labels": [],
    }
    samples: list[TurnSample] = []
    if not replay.steps:
        return samples, debug
    for player_id in range(len(replay.steps[0])):
        if player_id not in selected:
            debug["skipped_loser_turns"].append({"episode_id": replay.episode_id, "player_id": player_id})
            continue
        indexed_frames = IndexedReplayFrames([replay.steps[i][player_id]["observation"] for i in range(len(replay.steps))])
        for row_index in range(1, len(replay.steps)):
            row = replay.steps[row_index][player_id]
            actions = row.get("action") or []
            if actions:
                extracted, unmatched, ambiguous, unknown_targets = extract_actions_for_turn(
                    replay, player_id, row_index, target_hit_only, indexed_frames
                )
                debug["unmatched_actions"].extend(unmatched)
                debug["ambiguous_matches"].extend(ambiguous)
                debug["unknown_target_labels"].extend(unknown_targets)
                debug["extracted_action_target_labels"].extend(
                    {"episode_id": replay.episode_id, "step": row_index, "player_id": player_id, **a} for a in extracted
                )
                if not extracted:
                    continue
            else:
                if not keep_noop:
                    continue
                extracted = []
            sample = build_turn_sample(
                replay.steps[row_index - 1][player_id]["observation"],
                player_id,
                replay.episode_id,
                row_index - 1,
                extracted,
                max_planets,
                max_fleets,
                max_actions,
                noop_stop_weight,
                winner_id=winner_id,
            )
            if sample is not None:
                samples.append(sample)
    return samples, debug
