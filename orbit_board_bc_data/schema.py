from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PLANET_FEATURES = (
    "relative_owner",
    "x",
    "y",
    "radius",
    "ships",
    "log_ships",
    "production",
    "is_mine",
    "is_enemy",
    "is_neutral",
    "is_comet",
    "is_orbiting",
    "orbital_radius",
    "orbital_phase_sin",
    "orbital_phase_cos",
    "velocity_x",
    "velocity_y",
)

FLEET_FEATURES = (
    "relative_owner",
    "x",
    "y",
    "angle_sin",
    "angle_cos",
    "from_planet_id",
    "ships",
    "log_ships",
    "speed",
    "is_friendly",
    "is_enemy",
)

GLOBAL_FEATURES = (
    "step_norm",
    "angular_velocity",
    "my_planet_count",
    "enemy_planet_count",
    "neutral_planet_count",
    "my_planet_ships",
    "enemy_planet_ships",
    "my_fleet_ships",
    "enemy_fleet_ships",
    "global_ship_advantage",
    "next_comet_spawn_delta",
    "planet_count",
    "fleet_count",
)


@dataclass(frozen=True)
class FeatureSchema:
    planet_features: Sequence[str]
    fleet_features: Sequence[str]
    global_features: Sequence[str]


DEFAULT_FEATURE_SCHEMA = FeatureSchema(PLANET_FEATURES, FLEET_FEATURES, GLOBAL_FEATURES)


@dataclass(frozen=True)
class Planet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float

    @classmethod
    def from_raw(cls, raw: Sequence[float]) -> "Planet":
        return cls(int(raw[0]), int(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]), float(raw[5]), float(raw[6]))


@dataclass(frozen=True)
class Fleet:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: float

    @classmethod
    def from_raw(cls, raw: Sequence[float]) -> "Fleet":
        return cls(int(raw[0]), int(raw[1]), float(raw[2]), float(raw[3]), float(raw[4]), int(raw[5]), float(raw[6]))

