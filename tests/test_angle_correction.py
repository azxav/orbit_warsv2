import math
import importlib.util
from pathlib import Path


def _load_submission_main():
    path = Path(__file__).resolve().parents[1] / "submission" / "main.py"
    spec = importlib.util.spec_from_file_location("submission_main_for_angle_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _planet(pid, owner, x, y, ships=20, radius=2.0, production=3):
    return [pid, owner, x, y, radius, ships, production]


def test_intercept_base_angle_keeps_static_target_angle():
    submission_main = _load_submission_main()

    obs = {
        "step": 12,
        "angular_velocity": 0.04,
        "planets": [
            _planet(1, 0, 80.0, 50.0, ships=100),
            _planet(2, -1, 2.0, 50.0),
        ],
        "initial_planets": [
            _planet(1, 0, 80.0, 50.0, ships=100),
            _planet(2, -1, 2.0, 50.0),
        ],
        "comet_planet_ids": [],
    }

    angle = submission_main._intercept_base_angle(obs, obs["planets"][0], obs["planets"][1], ships=40)

    assert angle == math.atan2(0.0, -78.0)


def test_intercept_base_angle_leads_orbiting_target():
    submission_main = _load_submission_main()

    source = _planet(1, 0, 80.0, 50.0, ships=100)
    target = _planet(2, -1, 50.0, 20.0, radius=2.0)
    obs = {
        "step": 7,
        "angular_velocity": 0.05,
        "planets": [source, target],
        "initial_planets": [
            source,
            _planet(2, -1, 50.0, 20.0, radius=2.0),
        ],
        "comet_planet_ids": [],
    }

    ships = 40
    current_angle = math.atan2(float(target[3]) - float(source[3]), float(target[2]) - float(source[2]))
    speed = submission_main._fleet_speed(ships)
    t_star = max(0.0, math.hypot(float(target[2]) - float(source[2]), float(target[3]) - float(source[3])) / speed)
    for _ in range(6):
        dt = max(1, min(submission_main._AIM_HORIZON, int(math.ceil(t_star))))
        phase = math.atan2(float(target[3]) - 50.0, float(target[2]) - 50.0) + float(obs["angular_velocity"]) * dt
        future_x = 50.0 + 30.0 * math.cos(phase)
        future_y = 50.0 + 30.0 * math.sin(phase)
        t_star = max(0.0, math.hypot(future_x - float(source[2]), future_y - float(source[3])) / speed)
    expected_angle = math.atan2(future_y - float(source[3]), future_x - float(source[2]))

    angle = submission_main._intercept_base_angle(obs, source, target, ships=ships)

    assert abs(angle - current_angle) > 0.1
    assert angle == expected_angle


def test_validated_launch_angle_hits_moving_target_when_base_misses():
    submission_main = _load_submission_main()

    source = _planet(1, 0, 80.0, 50.0, ships=100, radius=2.0)
    target = _planet(2, -1, 50.0, 20.0, ships=10, radius=1.5)
    obs = {
        "step": 20,
        "angular_velocity": 0.025,
        "planets": [source, target],
        "initial_planets": [source, target],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
    }

    bad_preference = -2.037352588344394
    angle = submission_main._validated_launch_angle(obs, source, target, ships=1, preferred_angle=bad_preference)
    assert angle is not None
    hit = submission_main._trace_launch(obs, source, angle, ships=1)

    assert hit is not None
    assert hit[0] == 2


def test_comet_position_does_not_fall_back_to_orbital_motion_after_path_end():
    submission_main = _load_submission_main()

    comet = _planet(35, -1, 20.0, 20.0, ships=10, radius=1.0)
    obs = {
        "step": 160,
        "angular_velocity": 0.1,
        "planets": [_planet(1, 0, 10.0, 10.0, ships=20), comet],
        "initial_planets": [_planet(1, 0, 10.0, 10.0, ships=20)],
        "comet_planet_ids": [35],
        "comets": [
            {
                "path_index": 1,
                "planet_ids": [35],
                "paths": [[[20.0, 20.0], [21.0, 21.0]]],
            }
        ],
    }

    assert submission_main._target_position_at_time(obs, comet, 0) == (21.0, 21.0)
    assert submission_main._target_position_at_time(obs, comet, 1) is None


def test_failed_validation_uses_bounded_search():
    submission_main = _load_submission_main()

    source = _planet(1, 0, 80.0, 50.0, ships=100, radius=2.0)
    target = _planet(2, -1, 50.0, 20.0, ships=10, radius=1.5)
    obs = {
        "step": 20,
        "angular_velocity": 0.025,
        "planets": [source, target],
        "initial_planets": [source, target],
        "fleets": [],
        "comets": [],
        "comet_planet_ids": [],
    }
    calls = {"trace": 0}

    def never_hits(*args, **kwargs):
        calls["trace"] += 1
        return None

    original_trace = submission_main._trace_launch
    submission_main._trace_launch = never_hits
    try:
        angle = submission_main._validated_launch_angle(obs, source, target, ships=1, preferred_angle=0.0)
    finally:
        submission_main._trace_launch = original_trace

    assert angle is None
    assert calls["trace"] <= 52
