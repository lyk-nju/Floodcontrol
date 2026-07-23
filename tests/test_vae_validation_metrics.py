import pytest
import torch

from utils.conditions.vae import BodyPrediction, VAEInput
from utils.motion_process import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    NUM_JOINTS,
)
from utils.training.vae.metrics import (
    _fk_joint_valid_mask,
    reconstruction_geometry_metrics,
)


def _geometry_case(frames: int = 8) -> tuple[VAEInput, BodyPrediction]:
    root = torch.zeros(1, frames, 5)
    root[..., 1] = 1.0
    root[..., 3] = 1.0
    body = torch.zeros(1, frames, BODY_DIM)
    identity_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    body[..., BODY_POSITION_DIM : BODY_POSITION_DIM + BODY_ROTATION_DIM] = (
        identity_rotation.repeat(NUM_JOINTS - 1)
    )
    frame_valid = torch.ones(1, frames, dtype=torch.bool)
    inputs = VAEInput(
        body_motion=body,
        root_motion=root,
        frame_valid_mask=frame_valid,
        body_feature_valid_mask=torch.ones_like(body, dtype=torch.bool),
    )
    prediction = BodyPrediction(
        continuous_body=body[..., :BODY_CONTINUOUS_DIM].clone(),
        contact_logits=torch.zeros(1, frames, 4),
    )
    return inputs, prediction


def test_validation_geometry_metrics_separate_direct_and_fk_errors():
    inputs, prediction = _geometry_case()
    perfect = reconstruction_geometry_metrics(inputs, prediction)
    assert set(perfect) == {
        "world_mpjpe_m",
        "source_fk_direct_mpjpe_m",
        "reconstruction_fk_direct_mpjpe_m",
        "reconstruction_fk_target_mpjpe_m",
    }
    assert all(value.item() == pytest.approx(0.0) for value in perfect.values())

    shifted = prediction.continuous_body.clone()
    shifted[..., :BODY_POSITION_DIM] += 1.0
    shifted_metrics = reconstruction_geometry_metrics(
        inputs,
        BodyPrediction(shifted, prediction.contact_logits),
    )
    expected = (21.0 / 22.0) * (3.0**0.5)
    assert shifted_metrics["world_mpjpe_m"].item() == pytest.approx(expected)
    assert shifted_metrics["reconstruction_fk_direct_mpjpe_m"].item() == pytest.approx(
        expected
    )
    assert shifted_metrics["source_fk_direct_mpjpe_m"].item() == pytest.approx(0.0)
    assert shifted_metrics["reconstruction_fk_target_mpjpe_m"].item() == pytest.approx(
        0.0
    )


def test_validation_geometry_metrics_ignore_right_padding():
    inputs, prediction = _geometry_case()
    inputs.frame_valid_mask[:, 4:] = False
    inputs.body_feature_valid_mask[:, 4:] = False
    prediction.continuous_body[:, 4:, :BODY_POSITION_DIM] = 100.0
    metrics = reconstruction_geometry_metrics(inputs, prediction)
    assert all(value.item() == pytest.approx(0.0) for value in metrics.values())


def test_fk_validity_requires_each_joint_cumulative_rotation():
    frame_valid = torch.ones(1, 2, dtype=torch.bool)
    rotation_valid = torch.ones(
        1, 2, NUM_JOINTS - 1, dtype=torch.bool
    )
    rotation_valid[..., 1] = False
    joint_valid = _fk_joint_valid_mask(frame_valid, rotation_valid)
    assert joint_valid[..., 0].all()
    assert joint_valid[..., 1].all()
    assert not joint_valid[..., 2].any()
