from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

from .geometry import wrap_angle
from .schema import Fleet


@dataclass(frozen=True)
class FleetMatch:
    fleet_id: int
    status: str
    fleet: Any


FLEET_ID = 0
FLEET_OWNER = 1
FLEET_ANGLE = 4
FLEET_SOURCE = 5
FLEET_SHIPS = 6


def match_created_fleet(
    action: list[float],
    before_obs: dict,
    after_obs: dict,
    player_id: int,
    angle_tolerance: float = 0.08,
    ship_tolerance: float = 1e-6,
    before_fleet_ids: set[int] | None = None,
    after_fleets: Iterable[Fleet] | None = None,
    after_fleet_rows: Iterable[list[float]] | None = None,
) -> FleetMatch | None:
    source_id = int(action[0])
    angle = float(action[1])
    ships = float(action[2])
    before_ids = before_fleet_ids if before_fleet_ids is not None else {int(f[0]) for f in before_obs.get("fleets", [])}
    raw_candidates: list[list[float]] = []
    if after_fleet_rows is not None:
        for fleet in after_fleet_rows:
            fleet_id = int(fleet[FLEET_ID])
            if fleet_id in before_ids:
                continue
            if int(fleet[FLEET_OWNER]) != player_id:
                continue
            if int(fleet[FLEET_SOURCE]) != source_id:
                continue
            if abs(float(fleet[FLEET_SHIPS]) - ships) > ship_tolerance:
                continue
            if abs(wrap_angle(float(fleet[FLEET_ANGLE]) - angle)) > angle_tolerance:
                continue
            raw_candidates.append(fleet)
        if len(raw_candidates) == 1:
            return FleetMatch(int(raw_candidates[0][FLEET_ID]), "matched", raw_candidates[0])
        if len(raw_candidates) > 1:
            return FleetMatch(int(raw_candidates[0][FLEET_ID]), "ambiguous", raw_candidates[0])
        return None

    candidates: list[Fleet] = []
    fleet_iter = after_fleets if after_fleets is not None else (Fleet.from_raw(raw) for raw in after_obs.get("fleets", []))
    for fleet in fleet_iter:
        if fleet.id in before_ids:
            continue
        if fleet.owner != player_id:
            continue
        if fleet.from_planet_id != source_id:
            continue
        if abs(fleet.ships - ships) > ship_tolerance:
            continue
        if abs(wrap_angle(fleet.angle - angle)) > angle_tolerance:
            continue
        candidates.append(fleet)
    if len(candidates) == 1:
        return FleetMatch(candidates[0].id, "matched", candidates[0])
    if len(candidates) > 1:
        return FleetMatch(candidates[0].id, "ambiguous", candidates[0])
    return None
