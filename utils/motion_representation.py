"""Physical root5/body265 representation and statistics utilities."""

from __future__ import annotations

import json
import hashlib
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
from utils.math.quaternion import cont6d_to_matrix, qinv, qrot, quaternion_to_matrix
from utils.local_frame import root_quat_to_physical_yaw
from utils.motion_process import recover_root_rot_pos


POSITION_SLICE = slice(0, BODY_POSITION_DIM)
ROTATION_SLICE = slice(BODY_POSITION_DIM, BODY_POSITION_DIM + BODY_ROTATION_DIM)
VELOCITY_SLICE = slice(BODY_POSITION_DIM + BODY_ROTATION_DIM, BODY_CONTINUOUS_DIM)
CONTACT_SLICE = slice(BODY_CONTINUOUS_DIM, BODY_DIM)

HUMANML_DIM = 263
HUMANML_SOURCE_REPRESENTATION = "humanml3d-263-ik-v1"
MOTION_CONVERTER_VERSION = "humanml265"
HUMANML_POSITION_SLICE = slice(4, 4 + BODY_POSITION_DIM)
HUMANML_ROTATION_SLICE = slice(
    HUMANML_POSITION_SLICE.stop,
    HUMANML_POSITION_SLICE.stop + (NUM_JOINTS - 1) * 6,
)
HUMANML_CONTACT_SLICE = slice(HUMANML_DIM - BODY_CONTACT_DIM, HUMANML_DIM)
HUMANML22_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16,
    17, 18, 19,
)


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


def humanml263_to_root_body_motion(
    motion: torch.Tensor,
    *,
    fps: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert processed HumanML3D features into the root5/body265 contract.

    HumanML3D stores heading-canonical joint positions, IK-derived local
    rotations for the 21 non-root joints, root deltas, heading-local joint
    velocities, and contacts. This conversion recovers physical positions and
    root heading, composes all 22 global rotations through the HumanML
    hierarchy, and recomputes global backward velocities in metres per second.

    ``MOTION_CONVERTER_VERSION == "humanml265"`` owns this complete algorithm:

    1. recover world root translation and physical heading from HumanML root
       deltas;
    2. rotate the 21 heading-canonical joint positions back to world space;
    3. compose the 21 IK-derived local rotations through ``HUMANML22_PARENTS``
       to obtain 22 global rotations;
    4. recompute all 22 world-space joint velocities with backward difference
       at the declared FPS, marking the cold-start velocity invalid; and
    5. retain the source binary foot contacts and pack root5/body265.

    Any mathematical change to one of these steps changes this single converter
    version. There are deliberately no separate position/rotation/velocity
    converter versions.
    """

    if motion.ndim == 2:
        motion = motion.unsqueeze(0)
        squeeze = True
    elif motion.ndim == 3:
        squeeze = False
    else:
        raise ValueError("HumanML3D motion must be [F,263] or [B,F,263]")
    if motion.shape[-1] != HUMANML_DIM:
        raise ValueError(f"HumanML3D motion must end in {HUMANML_DIM} features")
    if motion.shape[-2] < 1:
        raise ValueError("HumanML3D motion must contain at least one frame")

    motion = motion.float()
    canonical_heading, root_positions = recover_root_rot_pos(motion)
    # HumanML's stored IK rotations use ``canonical_heading`` itself at the
    # skeleton root (the official rotation-FK recovery follows this convention),
    # while world positions and physical facing use its inverse. Keep these two
    # meanings explicit instead of forcing root rotation and path heading equal.
    global_root_rotation = quaternion_to_matrix(canonical_heading)

    local_positions = motion[..., HUMANML_POSITION_SLICE].reshape(
        *motion.shape[:-1], NUM_JOINTS - 1, 3
    )
    global_positions_non_root = qrot(
        qinv(canonical_heading)[..., None, :].expand(
            *local_positions.shape[:-1], 4
        ),
        local_positions,
    )
    global_positions_non_root[..., 0] += root_positions[..., None, 0]
    global_positions_non_root[..., 2] += root_positions[..., None, 2]
    global_positions = torch.cat(
        [root_positions[..., None, :], global_positions_non_root], dim=-2
    )

    child_local_rotations = cont6d_to_matrix(
        motion[..., HUMANML_ROTATION_SLICE].reshape(
            *motion.shape[:-1], NUM_JOINTS - 1, 6
        )
    )
    global_rotations = [global_root_rotation]
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        global_rotations.append(
            global_rotations[parent] @ child_local_rotations[..., joint - 1, :, :]
        )
    global_rotations = torch.stack(global_rotations, dim=-3)
    heading = root_quat_to_physical_yaw(canonical_heading)
    contacts = motion[..., HUMANML_CONTACT_SLICE]
    root, body, feature_valid = build_root_body_motion(
        global_positions,
        global_rotations,
        root_positions,
        heading,
        fps=fps,
        foot_contacts=contacts,
    )
    if squeeze:
        return root[0], body[0], feature_valid[0]
    return root, body, feature_valid


def rotate_root_yaw(
    root_motion: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [B,F,5]")
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
    return root


def rotate_root_body_yaw(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [B,F,5]")
    if body_motion.ndim != 3 or body_motion.shape[-1] != BODY_DIM:
        raise ValueError(f"body_motion must be [B,F,{BODY_DIM}]")
    if root_motion.shape[:2] != body_motion.shape[:2]:
        raise ValueError("root_motion and body_motion must share [B,F]")
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
    root = rotate_root_yaw(root_motion, angle)
    parts = unpack_body_motion(body_motion)
    positions = torch.einsum("bij,bfkj->bfki", rotation, parts["joint_positions"])
    velocities = torch.einsum("bij,bfkj->bfki", rotation, parts["joint_velocities"])
    matrices = rotation_6d_to_matrix(parts["joint_rotations_6d"])
    matrices = torch.einsum("bij,bfkjl->bfkil", rotation, matrices)
    body = pack_body_motion(
        positions, matrix_to_rotation_6d(matrices), velocities, parts["foot_contacts"]
    )
    return root, body


def deterministic_sample_yaw(
    dataset: str,
    sample_id: str,
    *,
    seed: int = 0,
) -> float:
    """Map one namespaced sample to a stable uniform yaw in ``[0, 2π)``."""

    payload = f"{int(seed)}\0{dataset}\0{sample_id}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return float(integer / 2**64 * (2.0 * np.pi))


def motion_artifact_manifest_sha256(
    records: list[Mapping[str, object]],
    *,
    expected_fps: float,
) -> tuple[str, list[str]]:
    """Fingerprint the actual converted samples used by a statistics artifact.

    Split TXT contents alone are insufficient because an artifact can be rebuilt
    in place after its source motion changes. The digest therefore owns the
    dataset namespace, sample id, source hash, source representation, converter
    version and FPS of every referenced artifact. Paths are intentionally not
    hashed so an identical dataset remains portable between machines.
    """

    digest = hashlib.sha256()
    source_representations: set[str] = set()
    identities: list[tuple[str, str, Path]] = []
    for record in records:
        identities.append(
            (
                str(record.get("dataset", "")),
                str(record.get("name", "")),
                Path(record["artifact"]),
            )
        )
    identities.sort(key=lambda item: (item[0], item[1]))
    seen: set[tuple[str, str]] = set()
    for dataset, name, path in identities:
        identity = (dataset, name)
        if identity in seen:
            raise ValueError(f"duplicate motion artifact identity: {dataset}/{name}")
        seen.add(identity)
        with np.load(path, allow_pickle=False) as data:
            contract = str(np.asarray(data["contract_version"]).item())
            converter = str(np.asarray(data["converter_version"]).item())
            representation = str(np.asarray(data["source_representation"]).item())
            source_sha256 = str(np.asarray(data["source_sha256"]).item())
            fps = float(np.asarray(data["fps"]).item())
        if contract != CONTRACT_VERSION:
            raise ValueError(f"motion artifact contract version mismatch in {path}")
        if converter != MOTION_CONVERTER_VERSION:
            raise ValueError(f"motion artifact converter version mismatch in {path}")
        if not np.isclose(fps, float(expected_fps), rtol=0.0, atol=1e-6):
            raise ValueError(f"motion artifact FPS mismatch in {path}")
        source_representations.add(representation)
        for value in (dataset, name, source_sha256, representation, converter, repr(fps)):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest(), sorted(source_representations)


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
class VAEStatistics:
    """Statistics owned by the body tokenizer, excluding LDF global root."""

    local_root_mean: torch.Tensor
    local_root_std: torch.Tensor
    body_cont_mean: torch.Tensor
    body_cont_std: torch.Tensor
    metadata: Mapping[str, object]

    def validate(self) -> None:
        for name, value, dim in (
            ("local_root_mean", self.local_root_mean, 4),
            ("local_root_std", self.local_root_std, 4),
            ("body_cont_mean", self.body_cont_mean, BODY_CONTINUOUS_DIM),
            ("body_cont_std", self.body_cont_std, BODY_CONTINUOUS_DIM),
        ):
            if tuple(value.shape) != (dim,):
                raise ValueError(f"{name} must have shape [{dim}]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain only finite values")
            if name.endswith("std") and bool((value <= 0).any()):
                raise ValueError(f"{name} must be positive")

    def save(self, path: str | Path) -> None:
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            local_root_mean=self.local_root_mean.cpu().numpy(),
            local_root_std=self.local_root_std.cpu().numpy(),
            body_cont_mean=self.body_cont_mean.cpu().numpy(),
            body_cont_std=self.body_cont_std.cpu().numpy(),
            metadata=json.dumps(dict(self.metadata), sort_keys=True),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_fps: float | None = None,
    ) -> "VAEStatistics":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            arrays = [torch.from_numpy(data[name]).float() for name in (
                "local_root_mean", "local_root_std", "body_cont_mean",
                "body_cont_std"
            )]
        stats = cls(*arrays, metadata=metadata)
        stats.validate()
        if metadata.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("motion statistics contract version mismatch")
        if metadata.get("converter_version") != MOTION_CONVERTER_VERSION:
            raise ValueError("motion statistics converter version mismatch")
        if metadata.get("skeleton") != "humanml22-v1":
            raise ValueError("motion statistics skeleton version mismatch")
        if expected_fps is not None and not np.isclose(
            float(metadata.get("fps", float("nan"))),
            float(expected_fps),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("motion statistics FPS mismatch")
        return stats


__all__ = [
    "CONTACT_SLICE", "HUMANML22_PARENTS", "HUMANML_CONTACT_SLICE",
    "HUMANML_DIM", "HUMANML_POSITION_SLICE", "HUMANML_ROTATION_SLICE",
    "HUMANML_SOURCE_REPRESENTATION", "MOTION_CONVERTER_VERSION",
    "POSITION_SLICE", "ROTATION_SLICE", "VELOCITY_SLICE",
    "VAEStatistics", "backward_joint_velocities", "build_root_body_motion",
    "derive_patched_local_root", "detect_foot_contacts", "matrix_to_rotation_6d",
    "humanml263_to_root_body_motion", "pack_body_motion", "rotate_root_body_yaw",
    "rotate_root_yaw", "motion_artifact_manifest_sha256",
    "deterministic_sample_yaw",
    "rotation_6d_to_matrix", "unpack_body_motion",
]
