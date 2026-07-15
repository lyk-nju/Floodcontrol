"""Adapters from Floodcontrol physical motion to HumanML3D evaluation features."""

from __future__ import annotations

from typing import Literal

import torch

from utils.math.quaternion import qrot
from utils.motion_process import (
    BODY_DIM,
    NUM_JOINTS,
    ROOT_DIM,
    recover_joint_positions,
    recover_root_yaw,
    rotation_to_matrix,
    unpack_body,
)


HUMANML_DIM = 263
HUMANML22_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)


def _canonical_heading(root_yaw: torch.Tensor) -> torch.Tensor:
    """Return HumanML's world-to-heading quaternion for physical yaw."""

    half_angle = -0.5 * root_yaw
    quaternion = root_yaw.new_zeros(*root_yaw.shape, 4)
    quaternion[..., 0] = torch.cos(half_angle)
    quaternion[..., 2] = torch.sin(half_angle)
    return quaternion


def _matrix_to_cont6d(matrix: torch.Tensor) -> torch.Tensor:
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def _validate_physical_motion(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if root_motion.ndim not in (2, 3) or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [F,5] or [B,F,5]")
    if body_motion.ndim != root_motion.ndim or body_motion.shape[-1] != BODY_DIM:
        raise ValueError("body_motion must match root_motion and end in 265")
    if root_motion.shape[:-1] != body_motion.shape[:-1]:
        raise ValueError("root_motion and body_motion must share batch/frame shape")
    if root_motion.shape[-2] < 2:
        raise ValueError("HumanML evaluation conversion requires at least two frames")
    if not bool(torch.isfinite(root_motion).all()) or not bool(
        torch.isfinite(body_motion).all()
    ):
        raise ValueError("root_motion and body_motion must contain only finite values")
    squeeze = root_motion.ndim == 2
    if squeeze:
        root_motion = root_motion.unsqueeze(0)
        body_motion = body_motion.unsqueeze(0)
    return root_motion, body_motion, squeeze


def _child_local_rotations(global_rotations: torch.Tensor) -> torch.Tensor:
    local = []
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        local.append(
            global_rotations[..., parent, :, :].transpose(-1, -2)
            @ global_rotations[..., joint, :, :]
        )
    return torch.stack(local, dim=-3)


def convert_root5_body265_to_humanml263(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
    *,
    tail: Literal["drop", "approximate"] = "drop",
) -> torch.Tensor:
    """Convert physical root5/body265 to standard HumanML3D 263D features.

    HumanML stores the transition from pose ``t`` to pose ``t+1`` on row
    ``t``. Consequently ``F`` physical poses define exactly ``F-1`` rows. The
    default ``tail='drop'`` returns that exact representation. The optional
    ``tail='approximate'`` appends one row whose unavailable future transition
    is extrapolated from the final observed transition; it is intended only for
    length-preserving diagnostics, not canonical FID reporting.
    """

    if tail not in ("drop", "approximate"):
        raise ValueError("tail must be 'drop' or 'approximate'")
    root_motion, body_motion, squeeze = _validate_physical_motion(
        root_motion, body_motion
    )
    parts = unpack_body(body_motion)
    root_yaw = recover_root_yaw(root_motion)
    heading = _canonical_heading(root_yaw)
    world_positions = recover_joint_positions(root_motion, body_motion)

    # HumanML root transition channels use the next frame's heading for planar
    # displacement and half of the signed physical yaw delta with opposite sign.
    yaw_delta = torch.atan2(
        torch.sin(root_yaw[:, 1:] - root_yaw[:, :-1]),
        torch.cos(root_yaw[:, 1:] - root_yaw[:, :-1]),
    )
    root_angular = -0.5 * yaw_delta
    root_displacement = root_motion[:, 1:, :3] - root_motion[:, :-1, :3]
    root_local = qrot(heading[:, 1:], root_displacement)[..., [0, 2]]
    root_height = root_motion[:, :-1, 1:2]

    # body265 positions are world-axis, planar-root-relative. HumanML stores
    # the same positions rotated into its heading-canonical frame.
    relative_positions = parts["joint_positions"][:, :-1]
    ric = qrot(
        heading[:, :-1, None, :].expand(*relative_positions.shape[:-1], 4),
        relative_positions,
    ).flatten(-2)

    global_rotations = rotation_to_matrix(parts["joint_rotations"])
    local_rotations = _child_local_rotations(global_rotations)
    rotations = _matrix_to_cont6d(local_rotations[:, :-1]).flatten(-2)

    joint_displacement = world_positions[:, 1:] - world_positions[:, :-1]
    local_velocity = qrot(
        heading[:, :-1, None, :].expand(*joint_displacement.shape[:-1], 4),
        joint_displacement,
    ).flatten(-2)
    contacts = parts["foot_contacts"][:, :-1]

    exact = torch.cat(
        [
            root_angular[..., None],
            root_local,
            root_height,
            ric,
            rotations,
            local_velocity,
            contacts,
        ],
        dim=-1,
    )
    if tail == "approximate":
        final_relative = parts["joint_positions"][:, -1:]
        final_ric = qrot(
            heading[:, -1:, None, :].expand(*final_relative.shape[:-1], 4),
            final_relative,
        ).flatten(-2)
        final_rotations = _matrix_to_cont6d(local_rotations[:, -1:]).flatten(-2)
        final_joint_velocity = qrot(
            heading[:, -1:, None, :].expand(
                *joint_displacement[:, -1:].shape[:-1], 4
            ),
            joint_displacement[:, -1:],
        ).flatten(-2)
        final_root_local = qrot(
            heading[:, -1:], root_displacement[:, -1:]
        )[..., [0, 2]]
        approximate = torch.cat(
            [
                root_angular[:, -1:, None],
                final_root_local,
                root_motion[:, -1:, 1:2],
                final_ric,
                final_rotations,
                final_joint_velocity,
                parts["foot_contacts"][:, -1:],
            ],
            dim=-1,
        )
        exact = torch.cat([exact, approximate], dim=1)
    if exact.shape[-1] != HUMANML_DIM:
        raise AssertionError("HumanML adapter produced an invalid feature dimension")
    return exact[0] if squeeze else exact


__all__ = ["HUMANML_DIM", "convert_root5_body265_to_humanml263"]
