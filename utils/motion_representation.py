"""Physical root5/body265 representation and statistics utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from utils.conditions.ldf import derive_local_root_motion, project_root_heading
from utils.conditions.vae import (
    BODY_CONTACT_DIM,
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    CONTRACT_VERSION,
    FRAMES_PER_TOKEN,
    NUM_JOINTS,
    ROOT_DIM,
)


POSITION_SLICE = slice(0, BODY_POSITION_DIM)
ROTATION_SLICE = slice(BODY_POSITION_DIM, BODY_POSITION_DIM + BODY_ROTATION_DIM)
VELOCITY_SLICE = slice(BODY_POSITION_DIM + BODY_ROTATION_DIM, BODY_CONTINUOUS_DIM)
CONTACT_SLICE = slice(BODY_CONTINUOUS_DIM, BODY_DIM)


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-1] != 6:
        raise ValueError("rotation must end in six dimensions")
    first = F.normalize(rotation[..., :3], dim=-1)
    second = rotation[..., 3:]
    second = F.normalize(second - (first * second).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    if tuple(rotation.shape[-2:]) != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    return torch.cat([rotation[..., :, 0], rotation[..., :, 1]], dim=-1)


def unpack_body_motion(body_motion: torch.Tensor) -> dict[str, torch.Tensor]:
    if body_motion.shape[-1] != BODY_DIM:
        raise ValueError(f"body_motion must end in {BODY_DIM}")
    return {
        "joint_positions": body_motion[..., POSITION_SLICE].reshape(
            *body_motion.shape[:-1], NUM_JOINTS - 1, 3
        ),
        "joint_rotations_6d": body_motion[..., ROTATION_SLICE].reshape(
            *body_motion.shape[:-1], NUM_JOINTS, 6
        ),
        "joint_velocities": body_motion[..., VELOCITY_SLICE].reshape(
            *body_motion.shape[:-1], NUM_JOINTS, 3
        ),
        "foot_contacts": body_motion[..., CONTACT_SLICE],
    }


def pack_body_motion(
    joint_positions: torch.Tensor,
    joint_rotations_6d: torch.Tensor,
    joint_velocities: torch.Tensor,
    foot_contacts: torch.Tensor,
) -> torch.Tensor:
    prefix = joint_positions.shape[:-2]
    for value, tail, name in (
        (joint_positions, (NUM_JOINTS - 1, 3), "joint_positions"),
        (joint_rotations_6d, (NUM_JOINTS, 6), "joint_rotations_6d"),
        (joint_velocities, (NUM_JOINTS, 3), "joint_velocities"),
        (foot_contacts, (BODY_CONTACT_DIM,), "foot_contacts"),
    ):
        if value.shape[:-len(tail)] != prefix or tuple(value.shape[-len(tail):]) != tail:
            raise ValueError(f"{name} has incompatible shape {tuple(value.shape)}")
    return torch.cat(
        [joint_positions.flatten(-2), joint_rotations_6d.flatten(-2),
         joint_velocities.flatten(-2), foot_contacts], dim=-1
    )


def backward_joint_velocities(
    global_positions: torch.Tensor,
    *,
    fps: float = 20.0,
    previous_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if global_positions.ndim != 4 or tuple(global_positions.shape[-2:]) != (NUM_JOINTS, 3):
        raise ValueError("global_positions must be [B,F,22,3]")
    batch = global_positions.shape[0]
    if previous_positions is None:
        previous = global_positions[:, :1]
        cold = True
    else:
        if tuple(previous_positions.shape) != (batch, NUM_JOINTS, 3):
            raise ValueError("previous_positions must be [B,22,3]")
        previous = previous_positions[:, None]
        cold = False
    prior = torch.cat([previous, global_positions[:, :-1]], dim=1)
    velocity = (global_positions - prior) * float(fps)
    valid = torch.ones_like(velocity, dtype=torch.bool)
    if cold:
        velocity[:, 0] = 0
        valid[:, 0] = False
    return velocity, valid


def detect_foot_contacts(
    global_positions: torch.Tensor,
    velocities: torch.Tensor,
    *,
    foot_joint_indices: tuple[int, int, int, int] = (7, 10, 8, 11),
    height_threshold: float = 0.15,
    speed_threshold: float = 0.10,
) -> torch.Tensor:
    indices = torch.as_tensor(foot_joint_indices, device=global_positions.device)
    foot_pos = global_positions.index_select(-2, indices)
    foot_vel = velocities.index_select(-2, indices)
    return ((foot_pos[..., 1] < float(height_threshold)) &
            (foot_vel.norm(dim=-1) < float(speed_threshold))).to(global_positions.dtype)


def build_root_body_motion(
    global_positions: torch.Tensor,
    global_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    root_heading: torch.Tensor,
    *,
    fps: float = 20.0,
    previous_positions: torch.Tensor | None = None,
    foot_contacts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if tuple(global_rotations.shape[-3:]) != (NUM_JOINTS, 3, 3):
        raise ValueError("global_rotations must be [B,F,22,3,3]")
    if tuple(root_positions.shape) != (*global_positions.shape[:2], 3):
        raise ValueError("root_positions must be [B,F,3]")
    if tuple(root_heading.shape) != tuple(global_positions.shape[:2]):
        raise ValueError("root_heading must be [B,F]")
    root_motion = torch.cat(
        [root_positions, torch.cos(root_heading)[..., None], torch.sin(root_heading)[..., None]], dim=-1
    )
    local_positions = global_positions[..., 1:, :].clone()
    local_positions[..., 0] -= root_positions[..., None, 0]
    local_positions[..., 2] -= root_positions[..., None, 2]
    velocities, velocity_valid = backward_joint_velocities(
        global_positions, fps=fps, previous_positions=previous_positions
    )
    if foot_contacts is None:
        foot_contacts = detect_foot_contacts(global_positions, velocities)
    body = pack_body_motion(
        local_positions, matrix_to_rotation_6d(global_rotations), velocities, foot_contacts
    )
    feature_valid = torch.ones_like(body, dtype=torch.bool)
    feature_valid[..., VELOCITY_SLICE] = velocity_valid.flatten(-2)
    return project_root_heading(root_motion), body, feature_valid


def rotate_root_body_yaw(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if root_motion.ndim != 3 or body_motion.ndim != 3:
        raise ValueError("root_motion/body_motion must be [B,F,D]")
    batch = root_motion.shape[0]
    angle = torch.as_tensor(angle, device=root_motion.device, dtype=root_motion.dtype)
    if angle.ndim == 0:
        angle = angle.expand(batch)
    if tuple(angle.shape) != (batch,):
        raise ValueError("angle must be scalar or [B]")
    cos, sin = torch.cos(angle), torch.sin(angle)
    rotation = torch.zeros(batch, 3, 3, device=root_motion.device, dtype=root_motion.dtype)
    rotation[:, 0, 0], rotation[:, 0, 2] = cos, sin
    rotation[:, 1, 1] = 1
    rotation[:, 2, 0], rotation[:, 2, 2] = -sin, cos
    root = root_motion.clone()
    root[..., :3] = torch.einsum("bij,bfj->bfi", rotation, root[..., :3])
    heading = torch.atan2(root_motion[..., 4], root_motion[..., 3]) + angle[:, None]
    root[..., 3], root[..., 4] = torch.cos(heading), torch.sin(heading)
    parts = unpack_body_motion(body_motion)
    positions = torch.einsum("bij,bfkj->bfki", rotation, parts["joint_positions"])
    velocities = torch.einsum("bij,bfkj->bfki", rotation, parts["joint_velocities"])
    matrices = rotation_6d_to_matrix(parts["joint_rotations_6d"])
    matrices = torch.einsum("bij,bfkjl->bfkil", rotation, matrices)
    body = pack_body_motion(
        positions, matrix_to_rotation_6d(matrices), velocities, parts["foot_contacts"]
    )
    return root, body


def derive_patched_local_root(
    root_motion: torch.Tensor,
    previous_root_frame: torch.Tensor | None,
    *,
    fps: float = 20.0,
    previous_root_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [B,F,5]")
    if root_motion.shape[1] % FRAMES_PER_TOKEN:
        raise ValueError("root frame length must be divisible by four")
    patched = root_motion.reshape(root_motion.shape[0], -1, FRAMES_PER_TOKEN, ROOT_DIM)
    values, valid = derive_local_root_motion(patched, previous_root_frame, fps=fps)
    if previous_root_valid_mask is not None:
        if previous_root_frame is None:
            raise ValueError("previous_root_valid_mask requires previous_root_frame")
        if tuple(previous_root_valid_mask.shape) != (root_motion.shape[0],):
            raise ValueError("previous_root_valid_mask must be [B]")
        cold = ~previous_root_valid_mask.bool()
        values[cold, 0, 0, :3] = 0
        valid[cold, 0, 0, :3] = False
    return values, valid


@dataclass(frozen=True)
class MotionStatistics:
    global_root_mean: torch.Tensor
    global_root_std: torch.Tensor
    local_root_mean: torch.Tensor
    local_root_std: torch.Tensor
    body_cont_mean: torch.Tensor
    body_cont_std: torch.Tensor
    metadata: Mapping[str, object]

    def validate(self) -> None:
        for name, value, dim in (
            ("global_root_mean", self.global_root_mean, ROOT_DIM),
            ("global_root_std", self.global_root_std, ROOT_DIM),
            ("local_root_mean", self.local_root_mean, 4),
            ("local_root_std", self.local_root_std, 4),
            ("body_cont_mean", self.body_cont_mean, BODY_CONTINUOUS_DIM),
            ("body_cont_std", self.body_cont_std, BODY_CONTINUOUS_DIM),
        ):
            if tuple(value.shape) != (dim,):
                raise ValueError(f"{name} must have shape [{dim}]")
            if name.endswith("std") and bool((value <= 0).any()):
                raise ValueError(f"{name} must be positive")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            global_root_mean=self.global_root_mean.cpu().numpy(),
            global_root_std=self.global_root_std.cpu().numpy(),
            local_root_mean=self.local_root_mean.cpu().numpy(),
            local_root_std=self.local_root_std.cpu().numpy(),
            body_cont_mean=self.body_cont_mean.cpu().numpy(),
            body_cont_std=self.body_cont_std.cpu().numpy(),
            metadata=json.dumps(dict(self.metadata), sort_keys=True),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MotionStatistics":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            arrays = [torch.from_numpy(data[name]).float() for name in (
                "global_root_mean", "global_root_std", "local_root_mean",
                "local_root_std", "body_cont_mean", "body_cont_std"
            )]
        stats = cls(*arrays, metadata=metadata)
        stats.validate()
        if metadata.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("motion statistics contract version mismatch")
        return stats


__all__ = [
    "CONTACT_SLICE", "POSITION_SLICE", "ROTATION_SLICE", "VELOCITY_SLICE",
    "MotionStatistics", "backward_joint_velocities", "build_root_body_motion",
    "derive_patched_local_root", "detect_foot_contacts", "matrix_to_rotation_6d",
    "pack_body_motion", "rotate_root_body_yaw", "rotation_6d_to_matrix",
    "unpack_body_motion",
]
