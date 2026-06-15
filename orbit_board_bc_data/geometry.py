from __future__ import annotations

import math

CENTER_X = 50.0
CENTER_Y = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
LOG_1000 = math.log(1000.0)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def angle_between(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.atan2(y2 - y1, x2 - x1)


def fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    ships = max(1.0, float(ships))
    ratio = min(1.0, math.log(ships) / LOG_1000)
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
    cx = ax + t * vx
    cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


def segment_hits_circle(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, radius: float) -> bool:
    expanded = radius + 1e-6
    if cx < min(ax, bx) - expanded or cx > max(ax, bx) + expanded:
        return False
    if cy < min(ay, by) - expanded or cy > max(ay, by) + expanded:
        return False
    return point_segment_distance(cx, cy, ax, ay, bx, by) <= expanded


def out_of_bounds(x: float, y: float, board_size: float = BOARD_SIZE) -> bool:
    return x < 0.0 or y < 0.0 or x > board_size or y > board_size
