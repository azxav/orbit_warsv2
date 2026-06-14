from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import orjson as _ORJSON
except ModuleNotFoundError:
    _ORJSON = None


@dataclass
class Replay:
    episode_id: str
    configuration: dict[str, Any]
    rewards: list[float]
    statuses: list[str]
    steps: list[list[dict[str, Any]]]
    path: Path | None = None


def _load_json_bytes(data: bytes) -> Any:
    if _ORJSON is not None:
        return _ORJSON.loads(data)
    return json.loads(data)


def load_replay(path: str | Path) -> Replay:
    replay_path = Path(path)
    raw = _load_json_bytes(replay_path.read_bytes())
    return Replay(
        episode_id=str(raw.get("id") or replay_path.stem),
        configuration=dict(raw.get("configuration") or {}),
        rewards=list(raw.get("rewards") or raw.get("final_rewards") or []),
        statuses=list(raw.get("statuses") or []),
        steps=list(raw.get("steps") or []),
        path=replay_path,
    )


def iter_replay_paths(replay_dir: str | Path) -> list[Path]:
    return sorted(Path(replay_dir).glob("*.json"))


def find_winner_id(replay: Replay) -> int:
    if not replay.rewards:
        raise ValueError(f"Replay {replay.episode_id} has no rewards")
    return max(range(len(replay.rewards)), key=lambda idx: replay.rewards[idx])
