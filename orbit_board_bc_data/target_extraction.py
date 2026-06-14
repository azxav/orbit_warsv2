from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .geometry import BOARD_SIZE, CENTER_X, CENTER_Y, SUN_RADIUS, fleet_speed, out_of_bounds, segment_hits_circle
from .schema import Fleet, Planet

TARGET_OUT = -1
TARGET_UNKNOWN = -2


@dataclass(frozen=True)
class TargetLabel:
    target_id: int
    status: str
    disappear_step: int | None = None


class FrameLookup(Protocol):
    def __len__(self) -> int: ...

    def fleet_by_id(self, index: int, fleet_id: int) -> Fleet | None: ...

    def planets(self, index: int) -> list[Planet]: ...


FLEET_ID = 0
FLEET_X = 2
FLEET_Y = 3
FLEET_ANGLE = 4
FLEET_SHIPS = 6
PLANET_ID = 0
PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4


def _fleet_row_by_id(frames: list[dict] | FrameLookup, index: int, fleet_id: int) -> list[float] | None:
    if hasattr(frames, "fleet_row_by_id"):
        return frames.fleet_row_by_id(index, fleet_id)
    for raw in frames[index].get("fleets", []):
        if int(raw[FLEET_ID]) == fleet_id:
            return raw
    return None


def _planet_rows(frames: list[dict] | FrameLookup, index: int) -> list[list[float]]:
    if hasattr(frames, "planet_rows"):
        return frames.planet_rows(index)
    return list(frames[index].get("planets", []))


def extract_target_for_fleet(
    fleet_id: int,
    frames: list[dict] | FrameLookup,
    start_index: int,
    ship_speed: float = 6.0,
    board_size: float = BOARD_SIZE,
    sun_radius: float = SUN_RADIUS,
) -> TargetLabel:
    last: list[float] | None = None
    last_index: int | None = None
    for idx in range(start_index, len(frames)):
        current = _fleet_row_by_id(frames, idx, fleet_id)
        if current is not None:
            last = current
            last_index = idx
            continue
        if last is None:
            return TargetLabel(TARGET_UNKNOWN, "unknown", idx)
        last_x = float(last[FLEET_X])
        last_y = float(last[FLEET_Y])
        last_angle = float(last[FLEET_ANGLE])
        speed = fleet_speed(float(last[FLEET_SHIPS]), ship_speed)
        bx = last_x + math.cos(last_angle) * speed
        by = last_y + math.sin(last_angle) * speed
        hits: set[int] = set()
        for frame_index in (max(start_index, idx - 1), idx):
            for p in _planet_rows(frames, frame_index):
                if segment_hits_circle(last_x, last_y, bx, by, float(p[PLANET_X]), float(p[PLANET_Y]), float(p[PLANET_RADIUS])):
                    hits.add(int(p[PLANET_ID]))
        hits = list(hits)
        if len(hits) == 1:
            return TargetLabel(hits[0], "hit", idx)
        if len(hits) > 1:
            return TargetLabel(TARGET_UNKNOWN, "ambiguous_hit", idx)
        if segment_hits_circle(last_x, last_y, bx, by, CENTER_X, CENTER_Y, sun_radius) or out_of_bounds(bx, by, board_size):
            return TargetLabel(TARGET_OUT, "out", idx)
        return TargetLabel(TARGET_UNKNOWN, "disappeared_without_hit", idx)
    if last_index is not None:
        return TargetLabel(TARGET_UNKNOWN, "still_active", last_index)
    return TargetLabel(TARGET_UNKNOWN, "never_seen", None)
