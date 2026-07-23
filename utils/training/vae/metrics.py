"""Geometry metrics used by VAE validation.

The VAE exposes both direct joint positions and cumulative joint rotations.
These metrics keep their errors separate so validation can distinguish an
ordinary position reconstruction error from a position/rotation consistency
problem that may justify an FK-based training loss.
"""

from __future__ import annotations

import torch

from utils.conditions.vae import (
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    NUM_JOINTS,
    BodyPrediction,
    VAEInput,
)
from utils.motion_process import (
    HUMANML22_PARENTS,
    forward_kinematics_heading_frame,
    infer_humanml_skeleton_offsets,
    recover_joint_positions,
    rotation_to_matrix,
)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _fk_joint_valid_mask(
    frame_valid_mask: torch.Tensor,
    rotation_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Propagate validity through the cumulative-rotation FK chain."""

    valid = torch.zeros(
        *frame_valid_mask.shape,
        NUM_JOINTS,
        dtype=torch.bool,
        device=frame_valid_mask.device,
    )
    valid[..., 0] = frame_valid_mask
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        valid[..., joint] = (
            valid[..., parent] & rotation_valid_mask[..., joint - 1]
        )
    return valid


def reconstruction_geometry_metrics(
    inputs: VAEInput,
    reconstructed_body: BodyPrediction,
) -> dict[str, torch.Tensor]:
    """Measure deterministic reconstruction geometry in metres.

    The four returned values have deliberately different meanings:

    - ``world_mpjpe_m`` compares the reconstructed direct position channels to
      the target positions in world space.
    - ``source_fk_direct_mpjpe_m`` is the dataset representation's own
      FK/direct mismatch and therefore the baseline for the remaining FK
      metrics.
    - ``reconstruction_fk_direct_mpjpe_m`` measures internal consistency
      between reconstructed rotations and reconstructed direct positions.
    - ``reconstruction_fk_target_mpjpe_m`` measures FK positions reconstructed
      from predicted rotations against the target direct positions.

    Root joints are included to match the offline VAE evaluator.  Their error
    is zero because root5 is an authoritative decoder condition.
    """

    inputs.validate()
    reconstructed_body.validate()
    if reconstructed_body.continuous_body.shape[:2] != inputs.body_motion.shape[:2]:
        raise ValueError("reconstructed body and VAE input must share [B,F]")

    # Validation can run under bf16 autocast.  Geometry metrics are diagnostic,
    # so compute them in float32 for stable metre-scale reporting.
    root = inputs.root_motion.float()
    target = inputs.body_motion.float()
    reconstructed = reconstructed_body.continuous_body.float()
    frame_valid = inputs.frame_valid_mask
    feature_valid = frame_valid[..., None].expand_as(inputs.body_motion)
    if inputs.body_feature_valid_mask is not None:
        feature_valid = feature_valid & inputs.body_feature_valid_mask

    position_valid = feature_valid[..., :BODY_POSITION_DIM].reshape(
        *target.shape[:2], NUM_JOINTS - 1, 3
    ).all(dim=-1)
    joint_position_valid = torch.cat(
        [frame_valid[..., None], position_valid],
        dim=-1,
    )
    rotation_end = BODY_POSITION_DIM + BODY_ROTATION_DIM
    rotation_valid = feature_valid[..., BODY_POSITION_DIM:rotation_end].reshape(
        *target.shape[:2], NUM_JOINTS - 1, 6
    ).all(dim=-1)

    target_joints = recover_joint_positions(root, target)
    reconstructed_joints = recover_joint_positions(root, reconstructed)

    # HumanML subject scale is inferred from one real pose.  Validation
    # windows are right-padded, but select the first fully valid pose rather
    # than relying on frame zero implicitly.
    reference_valid = frame_valid & position_valid.all(dim=-1)
    if not bool(reference_valid.any(dim=1).all()):
        raise ValueError(
            "VAE geometry metrics require one fully valid reference pose per sample"
        )
    reference_index = reference_valid.to(torch.int64).argmax(dim=1)
    batch_index = torch.arange(target.shape[0], device=target.device)
    skeleton_offsets = infer_humanml_skeleton_offsets(
        target_joints[batch_index, reference_index]
    )

    target_rotation = rotation_to_matrix(
        target[..., BODY_POSITION_DIM:rotation_end].reshape(
            *target.shape[:2], NUM_JOINTS - 1, 6
        )
    )
    reconstructed_rotation = rotation_to_matrix(
        reconstructed[..., BODY_POSITION_DIM:rotation_end].reshape(
            *reconstructed.shape[:2], NUM_JOINTS - 1, 6
        )
    )
    target_fk_joints = forward_kinematics_heading_frame(
        root,
        target_rotation,
        skeleton_offsets,
    )
    reconstructed_fk_joints = forward_kinematics_heading_frame(
        root,
        reconstructed_rotation,
        skeleton_offsets,
    )

    fk_valid = _fk_joint_valid_mask(frame_valid, rotation_valid)
    fk_comparison_valid = fk_valid & joint_position_valid
    return {
        "world_mpjpe_m": _masked_mean(
            (reconstructed_joints - target_joints).norm(dim=-1),
            joint_position_valid,
        ),
        "source_fk_direct_mpjpe_m": _masked_mean(
            (target_fk_joints - target_joints).norm(dim=-1),
            fk_comparison_valid,
        ),
        "reconstruction_fk_direct_mpjpe_m": _masked_mean(
            (reconstructed_fk_joints - reconstructed_joints).norm(dim=-1),
            fk_comparison_valid,
        ),
        "reconstruction_fk_target_mpjpe_m": _masked_mean(
            (reconstructed_fk_joints - target_joints).norm(dim=-1),
            fk_comparison_valid,
        ),
    }


__all__ = ["reconstruction_geometry_metrics"]
