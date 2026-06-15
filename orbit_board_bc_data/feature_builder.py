from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import CENTER_X, CENTER_Y, ROTATION_RADIUS_LIMIT, fleet_speed
from .schema import DEFAULT_FEATURE_SCHEMA

LOG_1001 = math.log(1001.0)
PLANET_ID = 0
PLANET_OWNER = 1
PLANET_X = 2
PLANET_Y = 3
PLANET_RADIUS = 4
PLANET_SHIPS = 5
PLANET_PRODUCTION = 6
FLEET_ID = 0
FLEET_OWNER = 1
FLEET_X = 2
FLEET_Y = 3
FLEET_ANGLE = 4
FLEET_SOURCE = 5
FLEET_SHIPS = 6


@dataclass
class BoardFeatures:
    planet_tokens: np.ndarray
    fleet_tokens: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    fleet_mask: np.ndarray
    planet_ids: list[int]
    fleet_ids: list[int]


def _next_comet_spawn_delta(step: int) -> float:
    spawns = [50, 150, 250, 350, 450]
    future = [s for s in spawns if s >= step]
    if not future:
        return 1.0
    return (future[0] - step) / 500.0


def _planet_owner_values(owner: int, player_id: int) -> tuple[int, int, int, float]:
    if owner == player_id:
        return 1, 0, 0, 1.0
    if owner == -1:
        return 0, 0, 1, 0.0
    return 0, 1, 0, -1.0


def _fleet_owner_values(owner: int, player_id: int) -> tuple[int, int, float]:
    if owner == player_id:
        return 1, 0, 1.0
    return 0, 1, -1.0


def build_board_features(obs: dict, player_id: int, max_planets: int, max_fleets: int, max_speed: float = 6.0) -> BoardFeatures:
    planet_dim = len(DEFAULT_FEATURE_SCHEMA.planet_features)
    fleet_dim = len(DEFAULT_FEATURE_SCHEMA.fleet_features)
    global_dim = len(DEFAULT_FEATURE_SCHEMA.global_features)
    planet_tokens = np.zeros((max_planets, planet_dim), dtype=np.float32)
    fleet_tokens = np.zeros((max_fleets, fleet_dim), dtype=np.float32)
    planet_mask = np.zeros((max_planets,), dtype=bool)
    fleet_mask = np.zeros((max_fleets,), dtype=bool)
    planet_ids: list[int] = []
    fleet_ids: list[int] = []

    comet_ids = {int(pid) for pid in obs.get("comet_planet_ids", [])}
    planets = obs.get("planets", [])[:max_planets]
    angular_velocity = float(obs.get("angular_velocity", 0.0))
    my_planet_count = 0
    enemy_planet_count = 0
    neutral_planet_count = 0
    my_planet_ships = 0.0
    enemy_planet_ships = 0.0
    for i, p in enumerate(planets):
        pid = int(p[PLANET_ID])
        owner = int(p[PLANET_OWNER])
        x = float(p[PLANET_X])
        y = float(p[PLANET_Y])
        radius = float(p[PLANET_RADIUS])
        ships = float(p[PLANET_SHIPS])
        production = float(p[PLANET_PRODUCTION])
        is_mine, is_enemy, is_neutral, rel_owner = _planet_owner_values(owner, player_id)
        my_planet_count += is_mine
        enemy_planet_count += is_enemy
        neutral_planet_count += is_neutral
        if is_mine:
            my_planet_ships += ships
        elif is_enemy:
            enemy_planet_ships += ships
        dx = x - CENTER_X
        dy = y - CENTER_Y
        orbital_radius = math.hypot(dx, dy)
        phase = math.atan2(dy, dx)
        is_orbiting = 1 if orbital_radius + radius < ROTATION_RADIUS_LIMIT else 0
        phase_sin = math.sin(phase)
        phase_cos = math.cos(phase)
        velocity_x = -phase_sin * orbital_radius * angular_velocity * is_orbiting
        velocity_y = phase_cos * orbital_radius * angular_velocity * is_orbiting
        planet_tokens[i] = (
            rel_owner,
            x / 100.0,
            y / 100.0,
            radius / 10.0,
            ships / 1000.0,
            math.log1p(max(0.0, ships)) / LOG_1001,
            production / 5.0,
            is_mine,
            is_enemy,
            is_neutral,
            1 if pid in comet_ids else 0,
            is_orbiting,
            orbital_radius / 100.0,
            phase_sin,
            phase_cos,
            velocity_x / 10.0,
            velocity_y / 10.0,
        )
        planet_mask[i] = True
        planet_ids.append(pid)

    fleets = obs.get("fleets", [])[:max_fleets]
    my_fleet_ships = 0.0
    enemy_fleet_ships = 0.0
    for i, f in enumerate(fleets):
        fid = int(f[FLEET_ID])
        owner = int(f[FLEET_OWNER])
        x = float(f[FLEET_X])
        y = float(f[FLEET_Y])
        angle = float(f[FLEET_ANGLE])
        source_id = int(f[FLEET_SOURCE])
        ships = float(f[FLEET_SHIPS])
        is_friendly, is_enemy, rel_owner = _fleet_owner_values(owner, player_id)
        if is_friendly:
            my_fleet_ships += ships
        else:
            enemy_fleet_ships += ships
        speed = fleet_speed(ships, max_speed)
        fleet_tokens[i] = (
            rel_owner,
            x / 100.0,
            y / 100.0,
            math.sin(angle),
            math.cos(angle),
            float(source_id) / 1000.0,
            ships / 1000.0,
            math.log1p(max(0.0, ships)) / LOG_1001,
            speed / max_speed,
            is_friendly,
            is_enemy,
        )
        fleet_mask[i] = True
        fleet_ids.append(fid)

    my_ships = my_planet_ships + my_fleet_ships
    enemy_ships = enemy_planet_ships + enemy_fleet_ships
    step = int(obs.get("step", 0))
    global_features = np.array(
        [
            step / 500.0,
            angular_velocity,
            my_planet_count / max(1, max_planets),
            enemy_planet_count / max(1, max_planets),
            neutral_planet_count / max(1, max_planets),
            my_planet_ships / 5000.0,
            enemy_planet_ships / 5000.0,
            my_fleet_ships / 5000.0,
            enemy_fleet_ships / 5000.0,
            (my_ships - enemy_ships) / max(1.0, my_ships + enemy_ships),
            _next_comet_spawn_delta(step),
            len(planets) / max(1, max_planets),
            len(fleets) / max(1, max_fleets),
        ],
        dtype=np.float32,
    )
    return BoardFeatures(planet_tokens, fleet_tokens, global_features, planet_mask, fleet_mask, planet_ids, fleet_ids)
