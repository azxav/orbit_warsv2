import numpy as np
import torch


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

