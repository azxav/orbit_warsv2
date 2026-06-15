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
