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
            moves.append([int(mine[0]), math.atan2(float(target[3]) - float(mine[3]), float(target[2]) - float(mine[2])), ships])
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
        base_angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
        angle = base_angle + float(outputs["angle_offset"][0, j].item())
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
