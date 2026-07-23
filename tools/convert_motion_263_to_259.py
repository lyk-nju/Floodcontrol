"""Offline HumanML263 to Floodcontrol root5/body259 conversion.

HumanML 263D source layout (F: frame count):
    root half-angle delta                    [0]
    heading-local root XZ displacement      [1:3]
    root height                             [3]
    heading-canonical non-root positions    [4:67]     = [21, 3]
    IK-derived non-root local rotation6d    [67:193]   = [21, 6]
    heading-local joint velocities          [193:259]  = [22, 3]
    binary foot contacts                    [259:263]  = [4]

Floodcontrol target layout:
    root5  = [root_x, root_y, root_z, cos(yaw), sin(yaw)]
    body259 = heading-local non-root positions [63]
            + heading-frame cumulative IK rotation6d [126]
            + current-heading-local backward velocity [66]
            + backward/current contacts [4]

This module is an offline source adapter. Runtime models and datasets consume
only the converted root5/body259 artifact and must not import this file.
"""

from __future__ import annotations

import torch

from utils.coordinate_transform import wrap_angle, yaw_to_matrix
from utils.motion_process import (
    BODY_CONTACT_DIM,
    BODY_POSITION_DIM,
    FOOT_JOINT_INDICES,
    HUMANML22_PARENTS,
    NUM_JOINTS,
    build_motion,
    compute_joint_velocities,
)
from utils.math.quaternion import (
    cont6d_to_matrix,
    qinv,
    qrot,
    quaternion_to_matrix,
)


HUMANML_DIM = 263
HUMANML_POSITION_SLICE = slice(4, 4 + BODY_POSITION_DIM)
HUMANML_ROTATION_SLICE = slice(
    HUMANML_POSITION_SLICE.stop,
    HUMANML_POSITION_SLICE.stop + (NUM_JOINTS - 1) * 6,
)
HUMANML_CONTACT_SLICE = slice(HUMANML_DIM - BODY_CONTACT_DIM, HUMANML_DIM)
def _root_quat_to_physical_yaw(root_quat: torch.Tensor) -> torch.Tensor:
    """Interpret HumanML's recovered root quaternion as physical yaw.

    ``recover_root_263`` returns ``[cos(a),0,sin(a),0]`` where ``a`` is the
    accumulated half-angle.  HumanML's path convention has the opposite sign
    from Floodcontrol's physical heading, hence ``yaw = -2a``.
    """
    qw = root_quat[..., 0]
    qy = root_quat[..., 2]
    yaw = -2.0 * torch.atan2(qy, qw)
    finite = torch.isfinite(qw) & torch.isfinite(qy)
    return wrap_angle(torch.where(finite, yaw, torch.zeros_like(yaw)))


def _validate_motion_263(motion: torch.Tensor) -> torch.Tensor:
    if motion.ndim not in (2, 3) or motion.shape[-1] != HUMANML_DIM:
        raise ValueError("HumanML3D motion must be [F,263] or [B,F,263]")
    if motion.shape[-2] < 1:
        raise ValueError("HumanML3D motion must contain at least one frame")
    if not motion.is_floating_point():
        motion = motion.float()
    if not bool(torch.isfinite(motion).all()):
        raise ValueError("HumanML3D motion must contain only finite values")
    return motion


def recover_root_263(
    motion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover HumanML canonical root quaternion and world translation.

    Args:
        motion: HumanML source motion ``[F,263]`` or ``[B,F,263]``.

    Returns:
        Canonical root quaternion ``[...,F,4]`` and world root translation
        ``[...,F,3]``. Frame zero is anchored at world XZ zero because HumanML
        stores each transition on its preceding frame.
    """
    motion = _validate_motion_263(motion)
    half_angle_delta = motion[..., 0]
    accumulated_half_angle = torch.zeros_like(half_angle_delta)
    accumulated_half_angle[..., 1:] = half_angle_delta[..., :-1]
    accumulated_half_angle = torch.cumsum(accumulated_half_angle, dim=-1)

    canonical_heading = motion.new_zeros(*motion.shape[:-1], 4)
    canonical_heading[..., 0] = torch.cos(accumulated_half_angle)
    canonical_heading[..., 2] = torch.sin(accumulated_half_angle)

    root_positions = motion.new_zeros(*motion.shape[:-1], 3)
    root_positions[..., 1:, [0, 2]] = motion[..., :-1, 1:3]
    root_positions = qrot(qinv(canonical_heading), root_positions)
    root_positions = torch.cumsum(root_positions, dim=-2)
    root_positions[..., 1] = motion[..., 3]
    return canonical_heading, root_positions


def recover_joint_positions_263(
    motion: torch.Tensor,
    *,
    canonical_heading: torch.Tensor | None = None,
    root_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recover HumanML heading-canonical positions into world space.

    Args:
        motion: HumanML source motion ``[F,263]`` or ``[B,F,263]``.
        canonical_heading: Optional result from :func:`recover_root_263`.
        root_positions: Optional result from :func:`recover_root_263`.

    Returns:
        World joint positions ``[...,F,22,3]``.
    """
    motion = _validate_motion_263(motion)
    if canonical_heading is None or root_positions is None:
        canonical_heading, root_positions = recover_root_263(motion)
    local_positions = motion[..., HUMANML_POSITION_SLICE].reshape(
        *motion.shape[:-1], NUM_JOINTS - 1, 3
    )
    non_root = qrot(
        qinv(canonical_heading)[..., None, :].expand(
            *local_positions.shape[:-1], 4
        ),
        local_positions,
    )
    non_root[..., 0] += root_positions[..., None, 0]
    non_root[..., 2] += root_positions[..., None, 2]
    return torch.cat([root_positions[..., None, :], non_root], dim=-2)


def recover_joint_rotations_263(
    motion: torch.Tensor,
    *,
    canonical_heading: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose HumanML IK-derived local rotations in the HumanML IK gauge.

    These hierarchical matrices preserve the source 263D rotation channels
    for reconstruction.  Their root convention follows HumanML's official IK
    recovery and is not the physical-facing convention used by root5 yaw.

    Args:
        motion: HumanML source motion ``[F,263]`` or ``[B,F,263]``.
        canonical_heading: Optional result from :func:`recover_root_263`.

    Returns:
        HumanML IK-gauge hierarchical matrices ``[...,F,22,3,3]``.
    """
    motion = _validate_motion_263(motion)
    if canonical_heading is None:
        canonical_heading, _ = recover_root_263(motion)

    # HumanML's official rotation-FK recovery uses canonical_heading itself at
    # the skeleton root, while physical path facing uses its inverse convention.
    root_rotation = quaternion_to_matrix(canonical_heading)
    child_local = cont6d_to_matrix(
        motion[..., HUMANML_ROTATION_SLICE].reshape(
            *motion.shape[:-1], NUM_JOINTS - 1, 6
        )
    )
    global_rotations = [root_rotation]
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        global_rotations.append(
            global_rotations[parent] @ child_local[..., joint - 1, :, :]
        )
    return torch.stack(global_rotations, dim=-3)


def detect_foot_contacts(
    global_positions: torch.Tensor,
    *,
    fps: float = 20.0,
    foot_joint_indices: tuple[int, int, int, int] = FOOT_JOINT_INDICES,
    height_threshold: float = 0.15,
    speed_threshold: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect causal backward/current contacts and their validity.

    Contact row ``t`` is derived from the transition ``t-1 -> t``.  At a true
    sequence start there is no preceding frame, so the first contact row is
    zero and invalid.
    """
    velocities, velocity_valid = compute_joint_velocities(global_positions, fps=fps)
    indices = torch.as_tensor(foot_joint_indices, device=global_positions.device)
    foot_positions = global_positions.index_select(-2, indices)
    foot_velocities = velocities.index_select(-2, indices)
    contact_valid = velocity_valid.index_select(-2, indices).all(dim=-1)
    contacts = (
        (foot_positions[..., 1] < float(height_threshold))
        & (foot_velocities.norm(dim=-1) < float(speed_threshold))
    ).to(global_positions.dtype)
    contacts = torch.where(contact_valid, contacts, torch.zeros_like(contacts))
    return contacts, contact_valid


def convert_motion_263_to_259(
    motion: torch.Tensor,
    *,
    fps: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert HumanML263 into physical root5/body259.

    Args:
        motion: HumanML source motion ``[F,263]`` or ``[B,F,263]``.
        fps: Source frame rate used to recompute global backward velocity.

    Returns:
        Root5, body259 and body feature validity. Unbatched input produces
        ``[F,5]``, ``[F,259]`` and ``[F,259]`` outputs.
    """
    motion = _validate_motion_263(motion)
    squeeze = motion.ndim == 2
    if squeeze:
        motion = motion.unsqueeze(0)

    canonical_heading, root_positions = recover_root_263(motion)
    global_positions = recover_joint_positions_263(
        motion,
        canonical_heading=canonical_heading,
        root_positions=root_positions,
    )
    humanml_cumulative_rotations = recover_joint_rotations_263(
        motion,
        canonical_heading=canonical_heading,
    )
    root_yaw = _root_quat_to_physical_yaw(canonical_heading)
    root_rotation = yaw_to_matrix(root_yaw)
    heading_frame_rotations = (
        root_rotation.transpose(-1, -2)[..., None, :, :]
        @ humanml_cumulative_rotations[..., 1:, :, :]
    )
    # HumanML row t stores the forward transition t -> t+1.  Body259 freezes
    # all dynamics to backward/current semantics, so shift the observable
    # labels by one frame.  HumanML's final forward transition has no matching
    # Body259 frame and is intentionally outside the exact inverse range.
    humanml_contacts = motion[..., HUMANML_CONTACT_SLICE]
    contacts = torch.zeros_like(humanml_contacts)
    contact_valid = torch.zeros_like(humanml_contacts, dtype=torch.bool)
    contacts[..., 1:, :] = humanml_contacts[..., :-1, :]
    contact_valid[..., 1:, :] = True
    root, body, feature_valid = build_motion(
        global_positions,
        heading_frame_rotations,
        root_positions,
        root_yaw,
        contacts,
        fps=fps,
        foot_contact_valid_mask=contact_valid,
    )
    if squeeze:
        return root[0], body[0], feature_valid[0]
    return root, body, feature_valid


__all__ = [
    "HUMANML22_PARENTS",
    "HUMANML_CONTACT_SLICE",
    "HUMANML_DIM",
    "HUMANML_POSITION_SLICE",
    "HUMANML_ROTATION_SLICE",
    "convert_motion_263_to_259",
    "detect_foot_contacts",
    "recover_joint_positions_263",
    "recover_joint_rotations_263",
    "recover_root_263",
]
