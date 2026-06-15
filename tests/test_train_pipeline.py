from argparse import Namespace

import numpy as np
import torch

from orbit_board_bc_data.sample_builder import TurnSample
from orbit_board_bc_data.tensor_writer import write_dataset
from orbit_board_bc_train.checkpoint import load_checkpoint


def test_collate_model_and_loss_smoke():
    from orbit_board_bc_train.collate import collate_samples
    from orbit_board_bc_train.losses import compute_board_bc_loss
    from orbit_board_bc_train.model.board_bc_model import BoardBCModel

    samples = []
    for _ in range(2):
        samples.append(
            {
                "planet_tokens": np.random.randn(4, 16).astype("float32"),
                "fleet_tokens": np.random.randn(3, 10).astype("float32"),
                "global_features": np.random.randn(13).astype("float32"),
                "planet_masks": np.array([1, 1, 0, 0], dtype=bool),
                "fleet_masks": np.array([1, 0, 0], dtype=bool),
                "action_source_labels": np.array([0, 4, 4], dtype="int64"),
                "action_target_labels": np.array([1, -100, -100], dtype="int64"),
                "action_angle_offset_labels": np.array([0.0, 0.0, 0.0], dtype="float32"),
                "action_ship_fraction_labels": np.array([0.5, 0.0, 0.0], dtype="float32"),
                "action_stop_labels": np.array([0, 1, 1], dtype=bool),
                "action_loss_weights": np.array([1.0, 1.0, 0.0], dtype="float32"),
                "action_valid_mask": np.array([1, 1, 0], dtype=bool),
                "source_candidate_mask": np.array(
                    [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=bool
                ),
                "target_candidate_mask": np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool),
            }
        )

    batch = collate_samples(samples)
    model = BoardBCModel(
        planet_dim=16,
        fleet_dim=10,
        global_dim=13,
        hidden_dim=32,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        max_actions=3,
        max_planets=4,
    )
    outputs = model(batch)
    assert outputs["source_logits"].shape == (2, 3, 5)
    assert outputs["target_logits"].shape == (2, 3, 4)
    loss, parts = compute_board_bc_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert parts["source_loss"] >= 0.0


def test_collate_omits_metadata_when_samples_do_not_have_it():
    from orbit_board_bc_train.collate import collate_samples

    samples = [
        {
            "planet_tokens": np.zeros((4, 16), dtype=np.float32),
            "fleet_tokens": np.zeros((3, 10), dtype=np.float32),
            "global_features": np.zeros((13,), dtype=np.float32),
        }
    ]

    batch = collate_samples(samples)

    assert "episode_id" not in batch
    assert "sample_step" not in batch


def _sample(i: int) -> TurnSample:
    rng = np.random.default_rng(i)
    max_planets = 4
    max_fleets = 3
    max_actions = 3
    source_labels = np.array([0, max_planets, max_planets], dtype=np.int64)
    target_labels = np.array([1, -100, -100], dtype=np.int64)
    return TurnSample(
        planet_tokens=rng.normal(size=(max_planets, 16)).astype("float32"),
        fleet_tokens=rng.normal(size=(max_fleets, 10)).astype("float32"),
        global_features=rng.normal(size=(13,)).astype("float32"),
        planet_masks=np.array([1, 1, 0, 0], dtype=bool),
        fleet_masks=np.array([1, 0, 0], dtype=bool),
        action_source_labels=source_labels,
        action_target_labels=target_labels,
        action_angle_labels=np.zeros((max_actions,), dtype=np.float32),
        action_angle_offset_labels=np.zeros((max_actions,), dtype=np.float32),
        action_ship_fraction_labels=np.array([0.5, 0.0, 0.0], dtype=np.float32),
        action_stop_labels=np.array([0, 1, 1], dtype=bool),
        action_loss_weights=np.array([1.0, 1.0, 0.0], dtype=np.float32),
        action_valid_mask=np.array([1, 1, 0], dtype=bool),
        source_candidate_mask=np.array(
            [[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [0, 0, 0, 0, 1]], dtype=bool
        ),
        target_candidate_mask=np.array([[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=bool),
        turn_type=1,
        phase_id=0,
        winner_id=0,
        episode_id=f"episode-{i}",
        sample_step=i,
        planet_ids=[0, 1],
    )


def _train_args(dataset, out_dir, epochs, resume=None):
    return Namespace(
        dataset=str(dataset),
        out_dir=str(out_dir),
        hidden_dim=32,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        dropout=0.0,
        batch_size=2,
        epochs=epochs,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=1.0,
        noop_stop_weight=0.35,
        device="cpu",
        shuffle_block_size=8,
        log_interval=1,
        log_file=None,
        resume=None if resume is None else str(resume),
    )


def test_train_resumes_from_last_completed_epoch(tmp_path):
    from orbit_board_bc_train.train_loop import train

    dataset = tmp_path / "dataset"
    write_dataset(dataset, [_sample(i) for i in range(4)], [_sample(100)], {}, {})

    out_dir = tmp_path / "run"
    train(_train_args(dataset, out_dir, epochs=1))
    first = load_checkpoint(out_dir / "last.pt")
    assert first["train_state"]["epoch"] == 1
    assert first["optimizer_state"]

    train(_train_args(dataset, out_dir, epochs=3, resume=out_dir / "last.pt"))

    resumed = load_checkpoint(out_dir / "last.pt")
    assert resumed["train_state"]["epoch"] == 3
    assert resumed["train_state"]["start_epoch"] == 2


def test_train_writes_progress_to_log_file(tmp_path, capsys):
    from orbit_board_bc_train.train_loop import train

    dataset = tmp_path / "dataset"
    write_dataset(dataset, [_sample(i) for i in range(4)], [_sample(100)], {}, {})

    out_dir = tmp_path / "run"
    train(_train_args(dataset, out_dir, epochs=1))

    log_text = (out_dir / "train.log").read_text(encoding="utf-8")
    stdout = capsys.readouterr().out
    assert "Starting training" in log_text
    assert "Epoch 1/1 batch 1/2" in log_text
    assert "Epoch 1/1 metrics" in log_text
    assert "Epoch 1/1 batch 1/2" in stdout


def test_dataset_memory_maps_arrays_by_default(tmp_path):
    from orbit_board_bc_train.dataset import BoardBCDataset

    dataset = tmp_path / "dataset"
    write_dataset(dataset, [_sample(i) for i in range(2)], [_sample(100)], {}, {})

    train_ds = BoardBCDataset(dataset, "train")

    assert isinstance(train_ds.arrays["planet_tokens"], np.memmap)
    assert isinstance(train_ds.arrays["fleet_tokens"], np.memmap)
    assert isinstance(train_ds.arrays["source_candidate_mask"], np.memmap)


def test_dataset_can_skip_metadata_index_for_training(tmp_path):
    from orbit_board_bc_train.dataset import BoardBCDataset

    dataset = tmp_path / "dataset"
    write_dataset(dataset, [_sample(i) for i in range(2)], [_sample(100)], {}, {})

    train_ds = BoardBCDataset(dataset, "train", include_metadata=False)
    item = train_ds[0]

    assert train_ds.index is None
    assert "episode_id" not in item
    assert "sample_step" not in item


def test_dataset_can_load_only_requested_arrays(tmp_path):
    from orbit_board_bc_train.dataset import BoardBCDataset

    dataset = tmp_path / "dataset"
    write_dataset(dataset, [_sample(i) for i in range(2)], [_sample(100)], {}, {})

    train_ds = BoardBCDataset(
        dataset,
        "train",
        include_metadata=False,
        array_keys=["planet_tokens", "planet_masks", "action_valid_mask"],
    )
    item = train_ds[0]

    assert set(train_ds.arrays) == {"planet_tokens", "planet_masks", "action_valid_mask"}
    assert set(item) == {"planet_tokens", "planet_masks", "action_valid_mask"}


def test_training_array_keys_exclude_unused_dataset_fields():
    from orbit_board_bc_train.train_loop import TRAIN_ARRAY_KEYS

    assert "action_angle_labels" not in TRAIN_ARRAY_KEYS
    assert "turn_type" not in TRAIN_ARRAY_KEYS
    assert "phase_id" not in TRAIN_ARRAY_KEYS
    assert "winner_id" not in TRAIN_ARRAY_KEYS
    assert "planet_tokens" in TRAIN_ARRAY_KEYS
    assert "action_loss_weights" in TRAIN_ARRAY_KEYS


def test_block_shuffle_sampler_visits_each_index_once():
    from orbit_board_bc_train.train_loop import BlockShuffleSampler

    sampler = BlockShuffleSampler(range(17), block_size=5)
    indices = list(sampler)

    assert len(indices) == 17
    assert sorted(indices) == list(range(17))
