from __future__ import annotations

import random
from collections.abc import Sequence

from .replay_loader import Replay


def select_players(replay: Replay, player_filter: str) -> list[int]:
    player_count = len(replay.steps[0]) if replay.steps else len(replay.rewards)
    ranked = sorted(range(player_count), key=lambda idx: replay.rewards[idx] if idx < len(replay.rewards) else -9999, reverse=True)
    if player_filter == "winner":
        return ranked[:1]
    if player_filter == "top2":
        return ranked[:2]
    if player_filter == "all":
        return list(range(player_count))
    raise ValueError(f"Unknown player filter: {player_filter}")


def split_episode_ids(episode_ids: Sequence[str], valid_ratio: float, seed: int = 13) -> tuple[set[str], set[str]]:
    ids = list(dict.fromkeys(episode_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if len(ids) <= 1:
        return set(ids), set()
    valid_count = max(1, int(round(len(ids) * valid_ratio)))
    valid = set(ids[:valid_count])
    train = set(ids[valid_count:])
    return train, valid


def phase_id(step: int) -> int:
    if step <= 150:
        return 0
    if step <= 350:
        return 1
    return 2

