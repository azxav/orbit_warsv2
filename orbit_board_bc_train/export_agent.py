from __future__ import annotations

import base64
from pathlib import Path


AGENT_TEMPLATE = r'''"""
Generated Orbit Wars submission agent.

The exported model is embedded for portability. If torch is unavailable or
loading fails, the agent falls back to a nearest-planet policy so the submission
remains runnable.
"""

import base64
import math
import os
import sys
import tempfile

MODEL_B64 = "__MODEL_B64__"
_MODEL = None
_CKPT = None
_LOAD_ERROR = None
_CENTER_X = 50.0
_CENTER_Y = 50.0
_ROTATION_RADIUS_LIMIT = 50.0
_LAUNCH_SURFACE_OFFSET = 0.1
_BOARD_SIZE = 100.0
_SUN_RADIUS = 10.0
_AIM_HORIZON = 120
_LOG_1000 = math.log(1000.0)


def _obs_get(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _fleet_speed(ships, max_speed=6.0):
    ships = max(1.0, float(ships))
    ratio = min(1.0, math.log(ships) / _LOG_1000)
    return 1.0 + (float(max_speed) - 1.0) * (ratio**1.5)


def _planet_by_id(rows):
    return {int(p[0]): p for p in rows or []}


def _path_position(path, position):
    if position < 0.0 or not path:
        return None
    lo = int(math.floor(position))
    hi = lo + 1
    if lo < 0 or lo >= len(path):
        return None
    x0 = float(path[lo][0])
    y0 = float(path[lo][1])
    if not (math.isfinite(x0) and math.isfinite(y0)):
        return None
    if hi >= len(path):
        return (x0, y0)
    x1 = float(path[hi][0])
    y1 = float(path[hi][1])
    if not (math.isfinite(x1) and math.isfinite(y1)):
        return (x0, y0)
    frac = position - lo
    return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)


def _comet_position_at_time(obs, target_id, t):
    comets = _obs_get(obs, "comets", [])
    if not isinstance(comets, list):
        return None
    for group in comets:
        if not isinstance(group, dict):
            continue
        ids = [int(pid) for pid in group.get("planet_ids", [])]
        if int(target_id) not in ids:
            continue
        comet_idx = ids.index(int(target_id))
        paths = group.get("paths", [])
        if comet_idx >= len(paths):
            return None
        path_index = float(group.get("path_index", -1))
        return _path_position(paths[comet_idx], path_index + float(t))
    return None


def _target_position_at_time(obs, target, t):
    target_id = int(target[0])
    comet_pos = _comet_position_at_time(obs, target_id, t)
    comet_ids = _obs_get(obs, "comet_planet_ids", []) or []
    if target_id in {int(pid) for pid in comet_ids}:
        return comet_pos
    if comet_pos is not None:
        return comet_pos

    x = float(target[2])
    y = float(target[3])
    radius = float(target[4])
    initial = _planet_by_id(_obs_get(obs, "initial_planets", [])).get(target_id, target)
    dx0 = float(initial[2]) - _CENTER_X
    dy0 = float(initial[3]) - _CENTER_Y
    orbital_radius = math.hypot(dx0, dy0)
    if orbital_radius <= 0.5 or orbital_radius + radius >= _ROTATION_RADIUS_LIMIT:
        return (x, y)

    current_phase = math.atan2(y - _CENTER_Y, x - _CENTER_X)
    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)
    phase = current_phase + angular_velocity * float(t)
    return (
        _CENTER_X + orbital_radius * math.cos(phase),
        _CENTER_Y + orbital_radius * math.sin(phase),
    )


def _segment_point_distance(px, py, ax, ay, bx, by):
    length_sq = (ax - bx) ** 2 + (ay - by) ** 2
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    u = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / length_sq
    u = max(0.0, min(1.0, u))
    return math.hypot(px - (ax + u * (bx - ax)), py - (ay + u * (by - ay)))


def _swept_pair_hit(ax, ay, bx, by, p0x, p0y, p1x, p1y, radius):
    d0x = ax - p0x
    d0y = ay - p0y
    dvx = (bx - ax) - (p1x - p0x)
    dvy = (by - ay) - (p1y - p0y)
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - float(radius) * float(radius)
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    root = math.sqrt(disc)
    return (-b + root) / (2.0 * a) >= 0.0 and (-b - root) / (2.0 * a) <= 1.0


def _angle_distance(a, b):
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def _trace_launch(obs, source, angle, ships, max_dt=_AIM_HORIZON, target_only=None):
    speed = _fleet_speed(ships)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    x = float(source[2]) + math.cos(angle) * (float(source[4]) + _LAUNCH_SURFACE_OFFSET)
    y = float(source[3]) + math.sin(angle) * (float(source[4]) + _LAUNCH_SURFACE_OFFSET)
    planets = [target_only] if target_only is not None else list(_obs_get(obs, "planets", []) or [])
    for dt in range(1, int(max_dt) + 1):
        nx = x + vx
        ny = y + vy
        for planet in planets:
            if planet is None:
                continue
            p0 = _target_position_at_time(obs, planet, dt - 1)
            p1 = _target_position_at_time(obs, planet, dt)
            if p0 is None or p1 is None:
                if target_only is not None:
                    return None
                continue
            p0x, p0y = p0
            p1x, p1y = p1
            if _swept_pair_hit(x, y, nx, ny, p0x, p0y, p1x, p1y, float(planet[4])):
                return (int(planet[0]), dt)
        if not (0.0 <= nx <= _BOARD_SIZE and 0.0 <= ny <= _BOARD_SIZE):
            return None
        if _segment_point_distance(_CENTER_X, _CENTER_Y, x, y, nx, ny) < _SUN_RADIUS:
            return None
        x = nx
        y = ny
    return None


def _intercept_base_angle(obs, source, target, ships, fp_iters=6):
    sx = float(source[2])
    sy = float(source[3])
    speed = max(1e-6, _fleet_speed(ships))
    initial_pos = _target_position_at_time(obs, target, 0.0)
    if initial_pos is None:
        return math.atan2(float(target[3]) - sy, float(target[2]) - sx)
    tx, ty = initial_pos
    t_star = max(0.0, math.hypot(tx - sx, ty - sy) / speed)
    for _ in range(int(fp_iters)):
        dt = max(1, min(_AIM_HORIZON, int(math.ceil(t_star))))
        future_pos = _target_position_at_time(obs, target, dt)
        if future_pos is None:
            break
        tx, ty = future_pos
        t_star = max(0.0, math.hypot(tx - sx, ty - sy) / speed)
    return math.atan2(ty - sy, tx - sx)


def _validated_launch_angle(obs, source, target, ships, preferred_angle=None):
    target_id = int(target[0])

    def hits_target(angle):
        hit = _trace_launch(obs, source, angle, ships)
        return hit is not None and int(hit[0]) == target_id

    base_angle = _intercept_base_angle(obs, source, target, ships)
    current_angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
    seeds = [base_angle, current_angle]
    if preferred_angle is not None:
        seeds.insert(0, float(preferred_angle))
    for angle in seeds:
        if hits_target(angle):
            return angle

    sx = float(source[2])
    sy = float(source[3])
    speed = max(1e-6, _fleet_speed(ships))
    preferred = float(preferred_angle) if preferred_angle is not None else base_angle
    centers = []
    for dt in range(1, _AIM_HORIZON + 1):
        future_pos = _target_position_at_time(obs, target, dt)
        if future_pos is None:
            break
        tx, ty = future_pos
        angle = math.atan2(ty - sy, tx - sx)
        eta = math.hypot(tx - sx, ty - sy) / speed
        centers.append((_angle_distance(angle, preferred) + 0.02 * abs(float(dt) - eta), angle))
    centers.sort(key=lambda item: item[0])

    target_hits = []
    seen_angles = set()
    search_phases = (
        (centers[:14], (0.0,)),
        (centers[:8], (0.025, -0.025)),
        (centers[:4], (0.06, -0.06, 0.12, -0.12)),
    )
    for phase_centers, offsets in search_phases:
        for _score, center_angle in phase_centers:
            for offset in offsets:
                angle = center_angle + offset
                angle_key = round((angle + math.pi) % (2.0 * math.pi), 4)
                if angle_key in seen_angles:
                    continue
                seen_angles.add(angle_key)
                target_hit = _trace_launch(obs, source, angle, ships, target_only=target)
                if target_hit is not None:
                    target_hits.append((_angle_distance(angle, preferred) + 0.001 * float(target_hit[1]), angle))
                    if len(target_hits) >= 10:
                        break
            if len(target_hits) >= 10:
                break
        if len(target_hits) >= 10:
            break

    for _score, angle in sorted(target_hits, key=lambda item: item[0]):
        if hits_target(angle):
            return angle
    return None


def _fallback_agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    moves = []
    my_planets = [p for p in raw_planets if int(p[1]) == player]
    targets = [p for p in raw_planets if int(p[1]) != player]
    for mine in my_planets:
        if not targets:
            break
        target = min(targets, key=lambda p: (float(p[2]) - float(mine[2])) ** 2 + (float(p[3]) - float(mine[3])) ** 2)
        ships = min(int(float(mine[5]) // 2), int(float(target[5])) + 1)
        if ships > 0:
            angle = _validated_launch_angle(obs, mine, target, ships)
            if angle is not None:
                moves.append([int(mine[0]), angle, ships])
    return moves


def _load_model_once():
    global _MODEL, _CKPT, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return _MODEL
    try:
        try:
            import torch
            from orbit_board_bc_train.model.board_bc_model import BoardBCModel
        except Exception:
            parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent not in sys.path:
                sys.path.insert(0, parent)
            import torch
            from orbit_board_bc_train.model.board_bc_model import BoardBCModel
        raw = base64.b64decode(MODEL_B64.encode("ascii"))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as fh:
            fh.write(raw)
            ckpt_path = fh.name
        _CKPT = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = _CKPT["config"]
        model = BoardBCModel(**cfg)
        model.load_state_dict(_CKPT["model_state"])
        model.eval()
        _MODEL = model
        return _MODEL
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        return None


def _build_batch(obs, cfg):
    import numpy as np
    import torch
    from orbit_board_bc_data.feature_builder import build_board_features

    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    if isinstance(obs, dict):
        obs_dict = obs
    else:
        obs_dict = {
            "player": player,
            "planets": raw_planets,
            "fleets": getattr(obs, "fleets", []),
            "angular_velocity": getattr(obs, "angular_velocity", 0.0),
            "comet_planet_ids": getattr(obs, "comet_planet_ids", []),
            "step": getattr(obs, "step", 0),
        }
    features = build_board_features(obs_dict, player, cfg["max_planets"], 256)
    max_actions = cfg["max_actions"]
    max_planets = cfg["max_planets"]
    source_mask = np.zeros((1, max_actions, max_planets + 1), dtype=bool)
    target_mask = np.zeros((1, max_actions, max_planets), dtype=bool)
    planet_by_id = {int(p[0]): p for p in raw_planets}
    for idx, pid in enumerate(features.planet_ids):
        p = planet_by_id.get(pid)
        if p is not None:
            if int(p[1]) == player and float(p[5]) > 0:
                source_mask[0, :, idx] = True
            target_mask[0, :, idx] = True
    source_mask[0, :, max_planets] = True
    return {
        "planet_tokens": torch.as_tensor(features.planet_tokens[None], dtype=torch.float32),
        "fleet_tokens": torch.as_tensor(features.fleet_tokens[None], dtype=torch.float32),
        "global_features": torch.as_tensor(features.global_features[None], dtype=torch.float32),
        "planet_masks": torch.as_tensor(features.planet_mask[None], dtype=torch.bool),
        "fleet_masks": torch.as_tensor(features.fleet_mask[None], dtype=torch.bool),
        "source_candidate_mask": torch.as_tensor(source_mask, dtype=torch.bool),
        "target_candidate_mask": torch.as_tensor(target_mask, dtype=torch.bool),
        "planet_ids": features.planet_ids,
    }


def _model_agent(obs):
    import torch
    model = _load_model_once()
    if model is None or _CKPT is None:
        return None
    cfg = _CKPT["config"]
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planet_by_id = {int(p[0]): p for p in raw_planets}
    batch = _build_batch(obs, cfg)
    planet_ids = batch.pop("planet_ids")
    with torch.no_grad():
        outputs = model(batch)
    stop_idx = cfg["max_planets"]
    remaining = {pid: float(planet_by_id[pid][5]) for pid in planet_ids if pid in planet_by_id}
    moves = []
    for j in range(cfg["max_actions"]):
        source_idx = int(outputs["source_logits"][0, j].argmax().item())
        if source_idx == stop_idx or source_idx >= len(planet_ids):
            break
        target_idx = int(outputs["target_logits"][0, j].argmax().item())
        if target_idx >= len(planet_ids):
            continue
        source_id = planet_ids[source_idx]
        target_id = planet_ids[target_idx]
        source = planet_by_id.get(source_id)
        target = planet_by_id.get(target_id)
        if source is None or target is None:
            continue
        available = int(max(0, remaining.get(source_id, 0)))
        if available <= 0:
            continue
        frac = float(outputs["ship_fraction"][0, j].clamp(0.05, 1.0).item())
        ships = max(1, min(available, int(round(available * frac))))
        base_angle = _intercept_base_angle(obs, source, target, ships)
        preferred_angle = base_angle + float(outputs["angle_offset"][0, j].item())
        angle = _validated_launch_angle(obs, source, target, ships, preferred_angle)
        if angle is None:
            continue
        moves.append([int(source_id), angle, ships])
        remaining[source_id] = available - ships
    return moves


def agent(obs):
    moves = _model_agent(obs)
    if moves:
        return moves
    return _fallback_agent(obs)
'''


def export_agent(checkpoint: str | Path, out: str | Path) -> None:
    ckpt_bytes = Path(checkpoint).read_bytes()
    encoded = base64.b64encode(ckpt_bytes).decode("ascii")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(AGENT_TEMPLATE.replace("__MODEL_B64__", encoded), encoding="utf-8")
