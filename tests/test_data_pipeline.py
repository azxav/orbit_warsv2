import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _planet(pid, owner, x, y, ships=20, radius=2.0, production=3):
    return [pid, owner, x, y, radius, ships, production]


def _fleet(fid, owner, x, y, angle, source, ships):
    return [fid, owner, x, y, angle, source, ships]


def _write_noop_replay(path: Path, episode_id: str, owner: int = 0) -> None:
    path.write_text(
        json.dumps(
            {
                "id": episode_id,
                "configuration": {"episodeSteps": 2, "shipSpeed": 6.0},
                "rewards": [1, -1, -1, -1],
                "statuses": ["DONE", "DONE", "DONE", "DONE"],
                "steps": [
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 0,
                                "planets": [_planet(1, owner, 10.0, 10.0)],
                                "fleets": [],
                            },
                            "action": [],
                            "status": "ACTIVE",
                        }
                        for player_id in range(4)
                    ],
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 1,
                                "planets": [_planet(1, owner, 10.0, 10.0)],
                                "fleets": [],
                            },
                            "action": [],
                            "status": "ACTIVE",
                        }
                        for player_id in range(4)
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )


def test_winner_filter_and_replay_loader(tmp_path):
    from orbit_board_bc_data.replay_loader import find_winner_id, load_replay
    from orbit_board_bc_data.split import select_players

    replay_path = tmp_path / "episode-1-replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "id": "episode-1",
                "configuration": {"episodeSteps": 2, "shipSpeed": 6.0},
                "rewards": [-1, 1, -1, -1],
                "statuses": ["DONE", "DONE", "DONE", "DONE"],
                "steps": [
                    [
                        {"observation": {"player": p, "planets": [], "fleets": []}, "action": [], "status": "ACTIVE"}
                        for p in range(4)
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )

    replay = load_replay(replay_path)
    assert replay.episode_id == "episode-1"
    assert find_winner_id(replay) == 1
    assert select_players(replay, "winner") == [1]
    assert select_players(replay, "top2") == [1, 0]
    assert select_players(replay, "all") == [0, 1, 2, 3]


def test_replay_loader_uses_fast_json_parser_when_available(tmp_path, monkeypatch):
    import orbit_board_bc_data.replay_loader as replay_loader

    calls = []

    class FastJson:
        @staticmethod
        def loads(data):
            calls.append(data)
            return json.loads(data)

    replay_path = tmp_path / "episode-fast-replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "id": "episode-fast",
                "configuration": {},
                "rewards": [1, -1],
                "statuses": ["DONE", "DONE"],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_loader, "_ORJSON", FastJson)

    replay = replay_loader.load_replay(replay_path)

    assert replay.episode_id == "episode-fast"
    assert len(calls) == 1
    assert isinstance(calls[0], bytes)


def test_replay_loader_falls_back_to_stdlib_json(tmp_path, monkeypatch):
    import orbit_board_bc_data.replay_loader as replay_loader

    replay_path = tmp_path / "episode-stdlib-replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "id": "episode-stdlib",
                "configuration": {},
                "rewards": [1, -1],
                "statuses": ["DONE", "DONE"],
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay_loader, "_ORJSON", None)

    replay = replay_loader.load_replay(replay_path)

    assert replay.episode_id == "episode-stdlib"
    assert replay.rewards == [1, -1]


def test_replay_loader_preserves_null_top_level_rewards(tmp_path):
    from orbit_board_bc_data.replay_loader import find_winner_id, load_replay

    replay_path = tmp_path / "episode-null-rewards-replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "id": "episode-null-rewards",
                "configuration": {},
                "rewards": [None, None, None, None],
                "statuses": ["DONE", "DONE", "DONE", "DONE"],
                "steps": [
                    [
                        {"observation": {"player": p, "planets": [], "fleets": []}, "action": [], "reward": None}
                        for p in range(4)
                    ],
                    [
                        {"observation": {"player": p, "planets": [], "fleets": []}, "action": [], "reward": reward}
                        for p, reward in enumerate([-1, -1, 1, -1])
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )

    replay = load_replay(replay_path)

    assert replay.rewards == [None, None, None, None]
    with pytest.raises(ValueError, match="non-numeric rewards"):
        find_winner_id(replay)


def test_find_winner_id_rejects_non_numeric_rewards():
    from orbit_board_bc_data.replay_loader import Replay, find_winner_id

    replay = Replay("episode-bad", {}, [None, None], [], [])

    with pytest.raises(ValueError, match="non-numeric rewards"):
        find_winner_id(replay)


def test_perspective_normalizes_raw_player_ids():
    from orbit_board_bc_data.perspective import fleet_relative_owner, planet_relative_owner

    assert planet_relative_owner(owner=3, player_id=3) == (1, 0, 0, 1.0)
    assert planet_relative_owner(owner=1, player_id=3) == (0, 1, 0, -1.0)
    assert planet_relative_owner(owner=-1, player_id=3) == (0, 0, 1, 0.0)
    assert fleet_relative_owner(owner=3, player_id=3) == (1, 0, 1.0)
    assert fleet_relative_owner(owner=0, player_id=3) == (0, 1, -1.0)


def test_fleet_matching_uses_created_fleet_from_next_row():
    from orbit_board_bc_data.fleet_matching import match_created_fleet

    source = _planet(5, 2, 10.0, 10.0, ships=25, radius=2.0)
    before = {"planets": [source], "fleets": [], "player": 2}
    after = {
        "planets": [source],
        "fleets": [_fleet(9, 2, 12.0, 10.0, 0.0, 5, 10)],
        "player": 2,
    }

    match = match_created_fleet([5, 0.0, 10], before, after, player_id=2)
    assert match is not None
    assert match.fleet_id == 9
    assert match.status == "matched"


def test_segment_hits_circle_rejects_far_circle_before_distance_math(monkeypatch):
    import orbit_board_bc_data.geometry as geometry

    def fail_distance(*_args):
        raise AssertionError("far circle should be rejected by bounding box")

    monkeypatch.setattr(geometry, "point_segment_distance", fail_distance)

    assert geometry.segment_hits_circle(0.0, 0.0, 1.0, 0.0, 100.0, 100.0, 2.0) is False


def test_fleet_speed_reuses_log_1000_constant(monkeypatch):
    import orbit_board_bc_data.geometry as geometry

    original_log = geometry.math.log
    calls = []

    def counting_log(value):
        calls.append(value)
        return original_log(value)

    monkeypatch.setattr(geometry.math, "log", counting_log)

    assert geometry.fleet_speed(64.0, 6.0) > 1.0
    assert calls == [64.0]


def test_feature_builder_reuses_orbital_phase_trig(monkeypatch):
    import orbit_board_bc_data.feature_builder as feature_builder

    original_sin = feature_builder.math.sin
    original_cos = feature_builder.math.cos
    sin_calls = []
    cos_calls = []

    def counting_sin(value):
        sin_calls.append(value)
        return original_sin(value)

    def counting_cos(value):
        cos_calls.append(value)
        return original_cos(value)

    monkeypatch.setattr(feature_builder.math, "sin", counting_sin)
    monkeypatch.setattr(feature_builder.math, "cos", counting_cos)
    obs = {
        "step": 0,
        "angular_velocity": 0.03,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 20.0)],
        "fleets": [],
        "player": 0,
    }

    features = feature_builder.build_board_features(obs, player_id=0, max_planets=4, max_fleets=2)

    assert features.planet_mask.tolist() == [True, False, False, False]
    assert len(sin_calls) == 1
    assert len(cos_calls) == 1


def test_feature_builder_inlines_relative_owner_hot_path(monkeypatch):
    import orbit_board_bc_data.feature_builder as feature_builder
    import orbit_board_bc_data.perspective as perspective

    def fail_planet_relative_owner(*_args):
        raise AssertionError("feature hot path should inline planet perspective")

    def fail_fleet_relative_owner(*_args):
        raise AssertionError("feature hot path should inline fleet perspective")

    monkeypatch.setattr(perspective, "planet_relative_owner", fail_planet_relative_owner)
    monkeypatch.setattr(perspective, "fleet_relative_owner", fail_fleet_relative_owner)
    obs = {
        "step": 0,
        "angular_velocity": 0.03,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 20.0), _planet(2, -1, 30.0, 40.0)],
        "fleets": [_fleet(3, 0, 11.0, 21.0, 0.0, 1, 8)],
        "player": 0,
    }

    features = feature_builder.build_board_features(obs, player_id=0, max_planets=4, max_fleets=3)

    assert features.planet_tokens[0, 7] == 1.0
    assert features.planet_tokens[1, 9] == 1.0
    assert features.fleet_tokens[0, 9] == 1.0


def test_target_extraction_detects_segment_planet_hit():
    from orbit_board_bc_data.target_extraction import extract_target_for_fleet

    frames = [
        {"planets": [_planet(1, -1, 20.0, 10.0, radius=2.0)], "fleets": [_fleet(7, 0, 10.0, 10.0, 0.0, 0, 1000)]},
        {"planets": [_planet(1, -1, 20.0, 10.0, radius=2.0)], "fleets": [_fleet(7, 0, 16.0, 10.0, 0.0, 0, 1000)]},
        {"planets": [_planet(1, 0, 20.0, 10.0, radius=2.0)], "fleets": []},
    ]

    label = extract_target_for_fleet(7, frames, start_index=0, ship_speed=6.0)
    assert label.target_id == 1
    assert label.status == "hit"


def test_target_extraction_uses_indexed_frame_lookup():
    from orbit_board_bc_data.replay_index import IndexedReplayFrames
    from orbit_board_bc_data.target_extraction import extract_target_for_fleet

    frames = []
    for step in range(12):
        fleets = [[fid, 1, float(step), float(fid), 0.0, 0, 2] for fid in range(100, 180)]
        if step < 5:
            fleets.append(_fleet(7, 0, float(step * 4), 10.0, 0.0, 0, 1000))
        frames.append({"planets": [_planet(1, -1, 24.0, 10.0, radius=2.0)], "fleets": fleets})

    indexed = IndexedReplayFrames(frames)
    label = extract_target_for_fleet(7, indexed, start_index=0, ship_speed=6.0)
    assert label.target_id == 1
    assert label.status == "hit"
    assert indexed.fleet_lookup_count == 6


def test_target_extraction_uses_indexed_raw_rows_without_entity_dataclasses(monkeypatch):
    from orbit_board_bc_data.replay_index import IndexedReplayFrames
    from orbit_board_bc_data.schema import Fleet, Planet
    from orbit_board_bc_data.target_extraction import extract_target_for_fleet

    def fail_from_raw(_raw):
        raise AssertionError("indexed target extraction should use raw rows")

    monkeypatch.setattr(Fleet, "from_raw", fail_from_raw)
    monkeypatch.setattr(Planet, "from_raw", fail_from_raw)
    frames = [
        {"planets": [_planet(1, -1, 20.0, 10.0, radius=2.0)], "fleets": [_fleet(7, 0, 10.0, 10.0, 0.0, 0, 1000)]},
        {"planets": [_planet(1, -1, 20.0, 10.0, radius=2.0)], "fleets": [_fleet(7, 0, 16.0, 10.0, 0.0, 0, 1000)]},
        {"planets": [_planet(1, 0, 20.0, 10.0, radius=2.0)], "fleets": []},
    ]

    label = extract_target_for_fleet(7, IndexedReplayFrames(frames), start_index=0, ship_speed=6.0)

    assert label.target_id == 1
    assert label.status == "hit"


def test_fleet_matching_uses_indexed_raw_rows_without_entity_dataclasses(monkeypatch):
    from orbit_board_bc_data.fleet_matching import match_created_fleet
    from orbit_board_bc_data.replay_index import IndexedReplayFrames
    from orbit_board_bc_data.schema import Fleet

    def fail_from_raw(_raw):
        raise AssertionError("indexed fleet matching should use raw rows")

    monkeypatch.setattr(Fleet, "from_raw", fail_from_raw)
    frames = [
        {"planets": [], "fleets": [_fleet(1, 0, 0.0, 0.0, 0.0, 5, 3)]},
        {"planets": [], "fleets": [_fleet(1, 0, 0.0, 0.0, 0.0, 5, 3), _fleet(9, 2, 12.0, 10.0, 0.0, 5, 10)]},
    ]
    indexed = IndexedReplayFrames(frames)

    match = match_created_fleet(
        [5, 0.0, 10],
        before_obs=frames[0],
        after_obs=frames[1],
        player_id=2,
        before_fleet_ids=indexed.fleet_ids(0),
        after_fleet_rows=indexed.fleet_rows(1),
    )

    assert match is not None
    assert match.fleet_id == 9
    assert match.status == "matched"


def test_feature_builder_outputs_stable_shapes():
    from orbit_board_bc_data.feature_builder import build_board_features
    from orbit_board_bc_data.schema import DEFAULT_FEATURE_SCHEMA

    obs = {
        "step": 25,
        "angular_velocity": 0.03,
        "comet_planet_ids": [4],
        "planets": [_planet(1, 2, 10.0, 20.0), _planet(4, -1, 60.0, 70.0, radius=1.0, production=1)],
        "fleets": [_fleet(3, 2, 11.0, 21.0, math.pi / 2, 1, 8)],
        "player": 2,
    }

    features = build_board_features(obs, player_id=2, max_planets=4, max_fleets=3)
    assert features.planet_tokens.shape == (4, len(DEFAULT_FEATURE_SCHEMA.planet_features))
    assert features.fleet_tokens.shape == (3, len(DEFAULT_FEATURE_SCHEMA.fleet_features))
    assert features.global_features.shape == (len(DEFAULT_FEATURE_SCHEMA.global_features),)
    assert features.planet_mask.tolist() == [True, True, False, False]
    assert features.fleet_mask.tolist() == [True, False, False]
    assert not np.isnan(features.planet_tokens).any()


def test_feature_builder_uses_raw_rows_without_entity_dataclasses(monkeypatch):
    import orbit_board_bc_data.feature_builder as feature_builder
    from orbit_board_bc_data.schema import Fleet, Planet

    def fail_from_raw(_raw):
        raise AssertionError("feature builder hot path should not allocate entity dataclasses")

    monkeypatch.setattr(Planet, "from_raw", fail_from_raw)
    monkeypatch.setattr(Fleet, "from_raw", fail_from_raw)
    obs = {
        "step": 25,
        "angular_velocity": 0.03,
        "comet_planet_ids": [4],
        "planets": [_planet(1, 2, 10.0, 20.0), _planet(4, -1, 60.0, 70.0, radius=1.0, production=1)],
        "fleets": [_fleet(3, 2, 11.0, 21.0, math.pi / 2, 1, 8)],
        "player": 2,
    }

    features = feature_builder.build_board_features(obs, player_id=2, max_planets=4, max_fleets=3)
    assert features.planet_ids == [1, 4]
    assert features.fleet_ids == [3]


def test_sample_builder_writes_stop_and_weights():
    from orbit_board_bc_data.sample_builder import build_turn_sample

    obs = {
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 10.0, ships=20), _planet(2, -1, 30.0, 10.0, ships=5)],
        "fleets": [],
        "player": 0,
    }
    actions = [{"source_id": 1, "target_id": 2, "angle": 0.0, "num_ships": 10}]

    sample = build_turn_sample(
        obs,
        player_id=0,
        episode_id="ep",
        sample_step=0,
        extracted_actions=actions,
        max_planets=4,
        max_fleets=2,
        max_actions=3,
        noop_stop_weight=0.35,
    )

    assert sample.action_source_labels.tolist() == [0, 4, 4]
    assert sample.action_target_labels.tolist()[0] == 1
    assert sample.action_stop_labels.tolist() == [False, True, True]
    assert sample.action_loss_weights.tolist() == [1.0, 1.0, 0.0]
    assert sample.action_ship_fraction_labels[0] == 0.5


def test_sample_builder_uses_raw_planet_rows_for_labels(monkeypatch):
    import orbit_board_bc_data.sample_builder as sample_builder
    from orbit_board_bc_data.schema import Planet

    def fail_from_raw(_raw):
        raise AssertionError("sample builder hot path should reuse raw planet rows")

    monkeypatch.setattr(Planet, "from_raw", fail_from_raw)
    obs = {
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 10.0, ships=20), _planet(2, -1, 30.0, 10.0, ships=5)],
        "fleets": [],
        "player": 0,
    }
    actions = [{"source_id": 1, "target_id": 2, "angle": 0.0, "num_ships": 10}]

    sample = sample_builder.build_turn_sample(
        obs,
        player_id=0,
        episode_id="ep",
        sample_step=0,
        extracted_actions=actions,
        max_planets=4,
        max_fleets=2,
        max_actions=3,
        noop_stop_weight=0.35,
    )

    assert sample is not None
    assert sample.action_ship_fraction_labels[0] == 0.5


def test_streaming_writer_flushes_chunks_and_preserves_dataset_contract(tmp_path):
    from orbit_board_bc_data.sample_builder import build_turn_sample
    from orbit_board_bc_data.tensor_writer import StreamingDatasetWriter

    out_dir = tmp_path / "dataset"
    writer = StreamingDatasetWriter(
        out_dir,
        max_planets=4,
        max_fleets=2,
        max_actions=3,
        chunk_size=2,
    )
    for idx in range(5):
        obs = {
            "step": idx,
            "angular_velocity": 0.0,
            "comet_planet_ids": [],
            "planets": [_planet(1, 0, 10.0, 10.0, ships=20), _planet(2, -1, 30.0, 10.0, ships=5)],
            "fleets": [],
            "player": 0,
        }
        sample = build_turn_sample(
            obs,
            player_id=0,
            episode_id=f"ep-{idx}",
            sample_step=idx,
            extracted_actions=[],
            max_planets=4,
            max_fleets=2,
            max_actions=3,
            noop_stop_weight=0.35,
        )
        assert sample is not None
        writer.add_samples("train", [sample])
    writer.finalize(debug={}, args={"seed": 13})

    assert np.load(out_dir / "train" / "planet_tokens.npy").shape == (5, 4, 17)
    assert np.load(out_dir / "train" / "action_source_labels.npy").shape == (5, 3)
    assert np.load(out_dir / "valid" / "planet_tokens.npy").shape == (0, 4, 17)
    masks = np.load(out_dir / "train" / "action_masks.npz")
    assert masks["action_valid_mask"].shape == (5, 3)
    assert (out_dir / "train" / "sample_index.parquet").exists()


def test_streaming_writer_can_write_uncompressed_masks_for_fast_builds(tmp_path):
    from orbit_board_bc_data.sample_builder import build_turn_sample
    from orbit_board_bc_data.tensor_writer import StreamingDatasetWriter
    from orbit_board_bc_train.dataset import BoardBCDataset

    out_dir = tmp_path / "dataset"
    writer = StreamingDatasetWriter(
        out_dir,
        max_planets=4,
        max_fleets=2,
        max_actions=3,
        chunk_size=1,
        compress_masks=False,
    )
    obs = {
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 10.0, ships=20), _planet(2, -1, 30.0, 10.0, ships=5)],
        "fleets": [],
        "player": 0,
    }
    sample = build_turn_sample(
        obs,
        player_id=0,
        episode_id="ep",
        sample_step=0,
        extracted_actions=[],
        max_planets=4,
        max_fleets=2,
        max_actions=3,
        noop_stop_weight=0.35,
    )
    assert sample is not None
    writer.add_samples("train", [sample])
    writer.finalize(debug={}, args={"seed": 13})

    with zipfile.ZipFile(out_dir / "train" / "action_masks.npz") as archive:
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}

    dataset = BoardBCDataset(out_dir, split="train")
    assert len(dataset) == 1
    assert dataset[0]["action_valid_mask"].shape == (3,)


def test_streaming_writer_writes_mask_sidecars_and_dataset_reads_them(tmp_path):
    from orbit_board_bc_data.sample_builder import build_turn_sample
    from orbit_board_bc_data.tensor_writer import StreamingDatasetWriter
    from orbit_board_bc_train.dataset import BoardBCDataset

    out_dir = tmp_path / "dataset"
    writer = StreamingDatasetWriter(out_dir, max_planets=4, max_fleets=2, max_actions=3, chunk_size=1)
    obs = {
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 10.0, ships=20)],
        "fleets": [],
        "player": 0,
    }
    sample = build_turn_sample(obs, 0, "ep", 0, [], 4, 2, 3, 0.35)
    assert sample is not None
    writer.add_samples("train", [sample])
    writer.finalize(debug={}, args={"seed": 13})
    (out_dir / "train" / "action_masks.npz").unlink()

    assert (out_dir / "train" / "action_valid_mask.npy").exists()
    assert (out_dir / "train" / "source_candidate_mask.npy").exists()
    assert (out_dir / "train" / "target_candidate_mask.npy").exists()
    dataset = BoardBCDataset(out_dir, "train")
    assert len(dataset) == 1
    assert dataset[0]["source_candidate_mask"].shape == (3, 5)


def test_dataset_reader_keeps_npz_mask_compatibility(tmp_path):
    from orbit_board_bc_data.sample_builder import build_turn_sample
    from orbit_board_bc_data.tensor_writer import StreamingDatasetWriter
    from orbit_board_bc_train.dataset import BoardBCDataset

    out_dir = tmp_path / "dataset"
    writer = StreamingDatasetWriter(out_dir, max_planets=4, max_fleets=2, max_actions=3, chunk_size=1)
    obs = {
        "step": 0,
        "angular_velocity": 0.0,
        "comet_planet_ids": [],
        "planets": [_planet(1, 0, 10.0, 10.0, ships=20)],
        "fleets": [],
        "player": 0,
    }
    sample = build_turn_sample(obs, 0, "ep", 0, [], 4, 2, 3, 0.35)
    assert sample is not None
    writer.add_samples("train", [sample])
    writer.finalize(debug={}, args={"seed": 13})
    for name in ["action_valid_mask.npy", "source_candidate_mask.npy", "target_candidate_mask.npy"]:
        (out_dir / "train" / name).unlink()

    dataset = BoardBCDataset(out_dir, "train")
    assert len(dataset) == 1
    assert dataset[0]["target_candidate_mask"].shape == (3, 4)


def test_cli_build_supports_parallel_replay_workers(tmp_path):
    from orbit_board_bc_data.cli import main
    from orbit_board_bc_data.validator import validate_dataset

    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    for idx in range(3):
        replay_path = replay_dir / f"episode-{idx}-replay.json"
        replay_path.write_text(
            json.dumps(
                {
                    "id": f"episode-{idx}",
                    "configuration": {"episodeSteps": 2, "shipSpeed": 6.0},
                    "rewards": [1, -1, -1, -1],
                    "statuses": ["DONE", "DONE", "DONE", "DONE"],
                    "steps": [
                        [
                            {
                                "observation": {
                                    "player": player_id,
                                    "step": 0,
                                    "planets": [_planet(1, 0, 10.0, 10.0)],
                                    "fleets": [],
                                },
                                "action": [],
                                "status": "ACTIVE",
                            }
                            for player_id in range(4)
                        ],
                        [
                            {
                                "observation": {
                                    "player": player_id,
                                    "step": 1,
                                    "planets": [_planet(1, 0, 10.0, 10.0)],
                                    "fleets": [],
                                },
                                "action": [],
                                "status": "ACTIVE",
                            }
                            for player_id in range(4)
                        ],
                    ],
                }
            ),
            encoding="utf-8",
        )

    out_dir = tmp_path / "dataset"
    main(
        [
            "build",
            "--replay-dir",
            str(replay_dir),
            "--out-dir",
            str(out_dir),
            "--keep-noop",
            "--valid-ratio",
            "0.34",
            "--max-planets",
            "4",
            "--max-fleets",
            "2",
            "--max-actions-per-turn",
            "3",
            "--writer-chunk-size",
            "1",
            "--workers",
            "2",
        ]
    )

    report = validate_dataset(out_dir)
    assert report["splits"]["train"]["samples"] == 2
    assert report["splits"]["valid"]["samples"] == 1


def test_cli_build_can_write_worker_shards_without_parent_sample_transfer(tmp_path):
    from orbit_board_bc_data.cli import main
    from orbit_board_bc_data.validator import validate_dataset

    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    for idx in range(3):
        replay_path = replay_dir / f"episode-{idx}-replay.json"
        replay_path.write_text(
            json.dumps(
                {
                    "id": f"episode-{idx}",
                    "configuration": {"episodeSteps": 2, "shipSpeed": 6.0},
                    "rewards": [1, -1, -1, -1],
                    "statuses": ["DONE", "DONE", "DONE", "DONE"],
                    "steps": [
                        [
                            {
                                "observation": {
                                    "player": player_id,
                                    "step": 0,
                                    "planets": [_planet(1, 0, 10.0, 10.0)],
                                    "fleets": [],
                                },
                                "action": [],
                                "status": "ACTIVE",
                            }
                            for player_id in range(4)
                        ],
                        [
                            {
                                "observation": {
                                    "player": player_id,
                                    "step": 1,
                                    "planets": [_planet(1, 0, 10.0, 10.0)],
                                    "fleets": [],
                                },
                                "action": [],
                                "status": "ACTIVE",
                            }
                            for player_id in range(4)
                        ],
                    ],
                }
            ),
            encoding="utf-8",
        )

    out_dir = tmp_path / "dataset"
    main(
        [
            "build",
            "--replay-dir",
            str(replay_dir),
            "--out-dir",
            str(out_dir),
            "--keep-noop",
            "--valid-ratio",
            "0.34",
            "--max-planets",
            "4",
            "--max-fleets",
            "2",
            "--max-actions-per-turn",
            "3",
            "--workers",
            "2",
            "--worker-output",
            "shard",
        ]
    )

    report = validate_dataset(out_dir)
    info = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert report["splits"]["train"]["samples"] == 2
    assert report["splits"]["valid"]["samples"] == 1
    assert info["args"]["worker_output"] == "shard"
    assert not (out_dir / "_worker_chunks").exists()


def test_cli_build_tracks_quality_counts_without_debug_csvs(tmp_path):
    from orbit_board_bc_data.cli import main

    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    replay_path = replay_dir / "episode-action-replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "id": "episode-action",
                "configuration": {"episodeSteps": 3, "shipSpeed": 6.0},
                "rewards": [1, -1, -1, -1],
                "statuses": ["DONE", "DONE", "DONE", "DONE"],
                "steps": [
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 0,
                                "planets": [_planet(1, 0, 10.0, 10.0, ships=2000), _planet(2, -1, 20.0, 10.0)],
                                "fleets": [],
                            },
                            "action": [],
                            "status": "ACTIVE",
                        }
                        for player_id in range(4)
                    ],
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 1,
                                "planets": [_planet(1, 0, 10.0, 10.0, ships=1000), _planet(2, -1, 20.0, 10.0)],
                                "fleets": [_fleet(9, 0, 16.0, 10.0, 0.0, 1, 1000)] if player_id == 0 else [],
                            },
                            "action": [[1, 0.0, 1000]] if player_id == 0 else [],
                            "status": "ACTIVE",
                        }
                        for player_id in range(4)
                    ],
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 2,
                                "planets": [_planet(1, 0, 10.0, 10.0, ships=1000), _planet(2, 0, 20.0, 10.0)],
                                "fleets": [],
                            },
                            "action": [],
                            "status": "ACTIVE",
                        }
                        for player_id in range(4)
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "dataset"
    main(
        [
            "build",
            "--replay-dir",
            str(replay_dir),
            "--out-dir",
            str(out_dir),
            "--target-hit-only",
            "--valid-ratio",
            "0.0",
            "--max-planets",
            "4",
            "--max-fleets",
            "4",
            "--max-actions-per-turn",
            "3",
            "--workers",
            "1",
        ]
    )

    info = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["stats"]["matched_actions"] == 1
    assert info["stats"]["unmatched_actions"] == 0
    assert info["stats"]["ambiguous_matches"] == 0


def test_cli_build_discards_invalid_reward_replays_and_tracks_metadata(tmp_path):
    from orbit_board_bc_data.cli import main

    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    _write_noop_replay(replay_dir / "valid.json", "valid")
    invalid_replay = replay_dir / "invalid.json"
    invalid_replay.write_text(
        json.dumps(
            {
                "id": "invalid",
                "configuration": {"episodeSteps": 2, "shipSpeed": 6.0},
                "rewards": [None, None, None, None],
                "statuses": ["DONE", "DONE", "DONE", "DONE"],
                "steps": [
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 0,
                                "planets": [_planet(1, 0, 10.0, 10.0)],
                                "fleets": [],
                            },
                            "action": [],
                            "reward": None,
                            "status": "ACTIVE",
                        }
                        for player_id in range(4)
                    ],
                    [
                        {
                            "observation": {
                                "player": player_id,
                                "step": 1,
                                "planets": [_planet(1, 0, 10.0, 10.0)],
                                "fleets": [],
                            },
                            "action": [],
                            "reward": reward,
                            "status": "DONE",
                        }
                        for player_id, reward in enumerate([1, -1, -1, -1])
                    ],
                ],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "dataset"
    main(
        [
            "build",
            "--replay-dir",
            str(replay_dir),
            "--out-dir",
            str(out_dir),
            "--keep-noop",
            "--valid-ratio",
            "0.0",
            "--max-planets",
            "4",
            "--max-fleets",
            "2",
            "--max-actions-per-turn",
            "3",
            "--workers",
            "2",
            "--write-debug",
        ]
    )

    info = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
    skipped = (out_dir / "debug" / "skipped_invalid_replays.csv").read_text(encoding="utf-8")

    assert info["stats"]["train_samples"] == 1
    assert info["stats"]["skipped_invalid_replays"] == 1
    assert "invalid" in skipped
    assert "non-numeric rewards" in skipped


def test_cli_build_append_adds_only_new_replays_and_keeps_existing_splits(tmp_path):
    import pandas as pd

    from orbit_board_bc_data.cli import main
    from orbit_board_bc_data.validator import validate_dataset

    initial_replays = tmp_path / "initial"
    initial_replays.mkdir()
    _write_noop_replay(initial_replays / "initial-train.json", "initial-train")
    _write_noop_replay(initial_replays / "initial-valid.json", "initial-valid")
    out_dir = tmp_path / "dataset"
    main(
        [
            "build",
            "--replay-dir",
            str(initial_replays),
            "--out-dir",
            str(out_dir),
            "--keep-noop",
            "--valid-ratio",
            "0.5",
            "--max-planets",
            "4",
            "--max-fleets",
            "2",
            "--max-actions-per-turn",
            "3",
            "--workers",
            "1",
        ]
    )
    old_train = set(pd.read_parquet(out_dir / "train" / "sample_index.parquet")["episode_id"])
    old_valid = set(pd.read_parquet(out_dir / "valid" / "sample_index.parquet")["episode_id"])
    old_count = validate_dataset(out_dir)["splits"]["train"]["samples"] + validate_dataset(out_dir)["splits"]["valid"]["samples"]

    new_replays = tmp_path / "new"
    new_replays.mkdir()
    _write_noop_replay(new_replays / "duplicate.json", next(iter(old_train | old_valid)))
    _write_noop_replay(new_replays / "new-a.json", "new-a")
    _write_noop_replay(new_replays / "new-b.json", "new-b")
    main(
        [
            "build",
            "--append",
            "--replay-dir",
            str(new_replays),
            "--out-dir",
            str(out_dir),
            "--keep-noop",
            "--valid-ratio",
            "0.5",
            "--max-planets",
            "4",
            "--max-fleets",
            "2",
            "--max-actions-per-turn",
            "3",
            "--workers",
            "1",
        ]
    )

    report = validate_dataset(out_dir)
    new_train = set(pd.read_parquet(out_dir / "train" / "sample_index.parquet")["episode_id"])
    new_valid = set(pd.read_parquet(out_dir / "valid" / "sample_index.parquet")["episode_id"])
    new_count = report["splits"]["train"]["samples"] + report["splits"]["valid"]["samples"]
    assert old_train <= new_train
    assert old_valid <= new_valid
    assert "new-a" in new_train | new_valid
    assert "new-b" in new_train | new_valid
    assert new_count == old_count + 2


def test_cli_build_append_no_new_replays_leaves_dataset_untouched(tmp_path):
    from orbit_board_bc_data.cli import main

    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    _write_noop_replay(replay_dir / "episode.json", "episode")
    out_dir = tmp_path / "dataset"
    args = [
        "build",
        "--replay-dir",
        str(replay_dir),
        "--out-dir",
        str(out_dir),
        "--keep-noop",
        "--valid-ratio",
        "0.0",
        "--max-planets",
        "4",
        "--max-fleets",
        "2",
        "--max-actions-per-turn",
        "3",
        "--workers",
        "1",
    ]
    main(args)
    before = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))

    main(["build", "--append", *args[1:]])

    after = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert after == before


def test_cli_build_append_rejects_shape_mismatch_without_replacing_dataset(tmp_path):
    from orbit_board_bc_data.cli import main

    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    _write_noop_replay(replay_dir / "episode.json", "episode")
    out_dir = tmp_path / "dataset"
    main(
        [
            "build",
            "--replay-dir",
            str(replay_dir),
            "--out-dir",
            str(out_dir),
            "--keep-noop",
            "--valid-ratio",
            "0.0",
            "--max-planets",
            "4",
            "--max-fleets",
            "2",
            "--max-actions-per-turn",
            "3",
            "--workers",
            "1",
        ]
    )
    before = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
    new_replays = tmp_path / "new-replays"
    new_replays.mkdir()
    _write_noop_replay(new_replays / "new-episode.json", "new-episode")

    with pytest.raises(SystemExit):
        main(
            [
                "build",
                "--append",
                "--replay-dir",
                str(new_replays),
                "--out-dir",
                str(out_dir),
                "--keep-noop",
                "--valid-ratio",
                "0.0",
                "--max-planets",
                "5",
                "--max-fleets",
                "2",
                "--max-actions-per-turn",
                "3",
                "--workers",
                "1",
            ]
        )

    after = json.loads((out_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert after == before
