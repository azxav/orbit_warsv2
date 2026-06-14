from __future__ import annotations


def planet_relative_owner(owner: int, player_id: int) -> tuple[int, int, int, float]:
    if owner == player_id:
        return 1, 0, 0, 1.0
    if owner == -1:
        return 0, 0, 1, 0.0
    return 0, 1, 0, -1.0


def fleet_relative_owner(owner: int, player_id: int) -> tuple[int, int, float]:
    if owner == player_id:
        return 1, 0, 1.0
    return 0, 1, -1.0

