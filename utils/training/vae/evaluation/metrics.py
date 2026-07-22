"""Reconstruction quality metrics for BodyVAE evaluation."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from utils.conditions.vae import (
    BODY_CONTINUOUS_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    NUM_JOINTS,
)
from utils.motion_process import (
    FOOT_JOINT_INDICES,
    LEFT_ANKLE_INDEX,
    LEFT_TOE_INDEX,
    RIGHT_ANKLE_INDEX,
    RIGHT_TOE_INDEX,
    forward_kinematics_heading_frame,
    infer_humanml_skeleton_offsets,
    recover_joint_positions,
    rotation_to_matrix,
)
from utils.token_frame import frame_count_to_token_count

from .reconstruction import MotionSample, ReconstructionResult


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(value)
    if not bool(expanded.any()):
        return 0.0
    return float(value[expanded].mean())


def _direction_metrics(
    predicted_joints: torch.Tensor,
    target_joints: torch.Tensor,
    joint_valid: torch.Tensor,
) -> tuple[float, float, float]:
    """Return foot direction error, reverse ratio and ankle-toe length MAE."""

    pairs = (
        (LEFT_ANKLE_INDEX, LEFT_TOE_INDEX),
        (RIGHT_ANKLE_INDEX, RIGHT_TOE_INDEX),
    )
    predicted = torch.stack(
        [predicted_joints[:, toe] - predicted_joints[:, ankle] for ankle, toe in pairs],
        dim=1,
    )
    target = torch.stack(
        [target_joints[:, toe] - target_joints[:, ankle] for ankle, toe in pairs],
        dim=1,
    )
    predicted_length = predicted.norm(dim=-1)
    target_length = target.norm(dim=-1)
    pair_valid = torch.stack(
        [joint_valid[:, ankle] & joint_valid[:, toe] for ankle, toe in pairs],
        dim=1,
    )
    valid = (
        pair_valid
        & (predicted_length > 1e-6)
        & (target_length > 1e-6)
    )
    cosine = (
        (predicted * target).sum(dim=-1)
        / (predicted_length * target_length).clamp_min(1e-12)
    ).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cosine))
    return (
        _masked_mean(angle, valid),
        _masked_mean((cosine < 0).float(), valid),
        _masked_mean((predicted_length - target_length).abs(), valid),
    )


def reconstruction_metrics(
    sample: MotionSample,
    result: ReconstructionResult,
) -> dict[str, float | int | str]:
    """Measure physical reconstruction, contacts, skating, and stream parity."""

    target = sample.body_motion
    predicted = result.streamed_body.continuous_body[0]
    feature_valid = sample.body_feature_valid_mask
    position_error = (
        predicted[:, :BODY_POSITION_DIM] - target[:, :BODY_POSITION_DIM]
    ).abs()
    velocity_start = BODY_POSITION_DIM + BODY_ROTATION_DIM
    velocity_error = (
        predicted[:, velocity_start:]
        - target[:, velocity_start:BODY_CONTINUOUS_DIM]
    ).abs()

    pred_rotation = rotation_to_matrix(
        predicted[:, BODY_POSITION_DIM:velocity_start].reshape(
            -1, NUM_JOINTS - 1, 6
        )
    )
    target_rotation = rotation_to_matrix(
        target[:, BODY_POSITION_DIM:velocity_start].reshape(
            -1, NUM_JOINTS - 1, 6
        )
    )
    relative = pred_rotation.transpose(-1, -2) @ target_rotation
    cosine = (
        (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    ).clamp(-1.0, 1.0)
    rotation_error = torch.rad2deg(torch.acos(cosine))
    rotation_valid = feature_valid[:, BODY_POSITION_DIM:velocity_start].reshape(
        -1, NUM_JOINTS - 1, 6
    ).all(-1)

    target_contact = target[:, BODY_CONTINUOUS_DIM:].bool()
    predicted_contact = result.streamed_body.contact_logits[0].sigmoid() >= 0.5
    true_positive = int((predicted_contact & target_contact).sum())
    false_positive = int((predicted_contact & ~target_contact).sum())
    false_negative = int((~predicted_contact & target_contact).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    foot_indices = torch.tensor(FOOT_JOINT_INDICES)
    reconstructed_joints = recover_joint_positions(sample.root_motion, predicted)
    target_joints = recover_joint_positions(sample.root_motion, target)
    position_joint_valid = feature_valid[:, :BODY_POSITION_DIM].reshape(
        -1, NUM_JOINTS - 1, 3
    ).all(dim=-1)
    joint_valid = torch.cat(
        [
            torch.ones(
                target.shape[0], 1, dtype=torch.bool, device=target.device
            ),
            position_joint_valid,
        ],
        dim=-1,
    )
    world_mpjpe = (
        reconstructed_joints - target_joints
    ).norm(dim=-1)

    # HumanML derives subject scale from a reference pose.  Using the target's
    # first pose supplies the same offsets to target and prediction, so the
    # source FK/direct mismatch remains visible instead of being attributed to
    # the VAE.
    skeleton_offsets = infer_humanml_skeleton_offsets(target_joints[0])
    target_fk_joints = forward_kinematics_heading_frame(
        sample.root_motion,
        target_rotation,
        skeleton_offsets,
    )
    reconstructed_fk_joints = forward_kinematics_heading_frame(
        sample.root_motion,
        pred_rotation,
        skeleton_offsets,
    )
    source_fk_error = (target_fk_joints - target_joints).norm(dim=-1)
    reconstructed_fk_error = (
        reconstructed_fk_joints - reconstructed_joints
    ).norm(dim=-1)
    reconstructed_fk_target_error = (
        reconstructed_fk_joints - target_joints
    ).norm(dim=-1)
    foot_direction_error, foot_reverse_ratio, ankle_toe_length_mae = (
        _direction_metrics(
            reconstructed_joints,
            target_joints,
            joint_valid,
        )
    )
    reconstructed_foot_positions = reconstructed_joints.index_select(1, foot_indices)
    position_foot_speed = predicted.new_zeros(target.shape[0], 4)
    position_foot_speed[1:] = (
        reconstructed_foot_positions[1:] - reconstructed_foot_positions[:-1]
    ).norm(dim=-1) * float(sample.fps)
    foot_position_valid = feature_valid[:, :BODY_POSITION_DIM].reshape(
        -1, NUM_JOINTS - 1, 3
    ).all(dim=-1).index_select(1, foot_indices - 1)
    transition_valid = torch.zeros_like(foot_position_valid)
    transition_valid[1:] = foot_position_valid[1:] & foot_position_valid[:-1]
    contact_valid = feature_valid[:, BODY_CONTINUOUS_DIM:]
    position_skating_valid = transition_valid & contact_valid

    predicted_velocity = predicted[:, velocity_start:].reshape(-1, NUM_JOINTS, 3)
    velocity_valid = feature_valid[:, velocity_start:BODY_CONTINUOUS_DIM].reshape(
        -1, NUM_JOINTS, 3
    ).all(dim=-1).index_select(1, foot_indices)
    contact_probability = result.streamed_body.contact_logits[0].sigmoid()
    predicted_velocity_speed = predicted_velocity.index_select(
        1, foot_indices
    ).norm(dim=-1)

    metrics = {
        "protocol": result.protocol,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "frames": int(target.shape[0]),
        "tokens": frame_count_to_token_count(target.shape[0]),
        "position_mae_m": _masked_mean(
            position_error, feature_valid[:, :BODY_POSITION_DIM]
        ),
        "world_mpjpe_m": _masked_mean(world_mpjpe, joint_valid),
        "source_fk_direct_mpjpe_m": _masked_mean(source_fk_error, joint_valid),
        "reconstruction_fk_direct_mpjpe_m": _masked_mean(
            reconstructed_fk_error, joint_valid
        ),
        "reconstruction_fk_target_mpjpe_m": _masked_mean(
            reconstructed_fk_target_error, joint_valid
        ),
        "foot_direction_error_deg": foot_direction_error,
        "foot_reverse_ratio": foot_reverse_ratio,
        "ankle_toe_length_mae_m": ankle_toe_length_mae,
        "velocity_mae_mps": _masked_mean(
            velocity_error, feature_valid[:, velocity_start:BODY_CONTINUOUS_DIM]
        ),
        "rotation_geodesic_deg": _masked_mean(rotation_error, rotation_valid),
        "contact_accuracy": float((predicted_contact == target_contact).float().mean()),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "gt_contact_position_skating_mps": _masked_mean(
            target_contact.float() * position_foot_speed,
            position_skating_valid,
        ),
        "predicted_contact_position_skating_mps": _masked_mean(
            contact_probability * position_foot_speed,
            position_skating_valid,
        ),
        "gt_contact_velocity_feature_mps": _masked_mean(
            target_contact.float() * predicted_velocity_speed,
            velocity_valid & contact_valid,
        ),
        "stream_offline_max_abs": result.stream_offline_max_abs,
    }
    if result.reference_stream_body is not None:
        reference = result.reference_stream_body.continuous_body[0]
        reference_position_error = (
            predicted[:, :BODY_POSITION_DIM] - reference[:, :BODY_POSITION_DIM]
        ).abs()
        reference_velocity_error = (
            predicted[:, velocity_start:] - reference[:, velocity_start:]
        ).abs()
        reference_rotation = rotation_to_matrix(
            reference[:, BODY_POSITION_DIM:velocity_start].reshape(
                -1, NUM_JOINTS - 1, 6
            )
        )
        rolling_relative = pred_rotation.transpose(-1, -2) @ reference_rotation
        rolling_cosine = (
            (rolling_relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        rolling_rotation_error = torch.rad2deg(torch.acos(rolling_cosine))
        reference_contacts = (
            result.reference_stream_body.contact_logits[0].sigmoid() >= 0.5
        )
        metrics.update(
            {
                "rolling_stream_position_mae_m": _masked_mean(
                    reference_position_error,
                    feature_valid[:, :BODY_POSITION_DIM],
                ),
                "rolling_stream_velocity_mae_mps": _masked_mean(
                    reference_velocity_error,
                    feature_valid[:, velocity_start:BODY_CONTINUOUS_DIM],
                ),
                "rolling_stream_rotation_geodesic_deg": _masked_mean(
                    rolling_rotation_error, rotation_valid
                ),
                "rolling_stream_contact_disagreement": float(
                    (predicted_contact != reference_contacts).float().mean()
                ),
                "rolling_stream_max_abs": float(result.rolling_reference_max_abs),
                "cache_window_offline_max_abs": result.stream_offline_max_abs,
            }
        )
    if result.rolling_trace is not None:
        metrics.update(
            {
                "history_tokens": int(result.rolling_trace["history_tokens"]),
                "commit_tokens": int(result.rolling_trace["commit_tokens"]),
                "rolling_steps": int(result.rolling_trace["commit_token"].shape[0]),
            }
        )
    return metrics


def mean_metrics(sample_metrics: list[Mapping[str, object]]) -> dict[str, float]:
    """Average scalar quality metrics while excluding structural counters."""

    structural = {
        "frames",
        "tokens",
        "history_tokens",
        "commit_tokens",
        "rolling_steps",
    }
    numeric_keys = sorted(
        key
        for key, value in sample_metrics[0].items()
        if isinstance(value, (float, int)) and key not in structural
    )
    return {
        key: float(np.mean([float(metrics[key]) for metrics in sample_metrics]))
        for key in numeric_keys
    }


__all__ = ["mean_metrics", "reconstruction_metrics"]
