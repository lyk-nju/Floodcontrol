"""Canonical physical root5/body259 motion processing.

Motion data structure (B: batch size, F: frame count):

root_motion [B, F, 5]
    root position xyz                         [0:3]
    physical heading cos(yaw), sin(yaw)      [3:5]

body_motion [B, F, 259]
    root-heading-local non-root positions [21, 3]       [0:63]
    heading-frame cumulative IK rotations [21, 6]      [63:189]
    current-heading-local backward velocities [22, 3]  [189:255]
    binary foot contacts                               [255:259]

Root5 is the only owner of world XYZ translation and absolute heading yaw.
Body259 is invariant to both world translation and a shared global yaw. The
rotation block inherits HumanML's IK gauge; it is not native SMPL rotation.
Normalization belongs to the VAE boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.coordinate_transform import (
    rotate_vectors_world_to_local,
    yaw_to_matrix,
)
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
    require_aligned_frame_count,
)


NUM_JOINTS = 22
HUMANML22_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
)
# Unit bone directions used by HumanML3D's 22-joint IK/FK contract.  Bone
# lengths remain sequence-specific and are inferred from a reference pose;
# keeping only directions here avoids silently imposing one subject's scale on
# every artifact.
HUMANML22_RAW_OFFSETS = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 0.0),
)
LEFT_ANKLE_INDEX = 7
RIGHT_ANKLE_INDEX = 8
LEFT_TOE_INDEX = 10
RIGHT_TOE_INDEX = 11
FOOT_JOINT_INDICES = (
    LEFT_ANKLE_INDEX,
    LEFT_TOE_INDEX,
    RIGHT_ANKLE_INDEX,
    RIGHT_TOE_INDEX,
)
ROOT_DIM = 5
LOCAL_ROOT_DIM = 4
BODY_POSITION_DIM = (NUM_JOINTS - 1) * 3
BODY_ROTATION_DIM = (NUM_JOINTS - 1) * 6
BODY_VELOCITY_DIM = NUM_JOINTS * 3
BODY_CONTINUOUS_DIM = BODY_POSITION_DIM + BODY_ROTATION_DIM + BODY_VELOCITY_DIM
BODY_CONTACT_DIM = 4
BODY_DIM = BODY_CONTINUOUS_DIM + BODY_CONTACT_DIM


POSITION_SLICE = slice(0, BODY_POSITION_DIM)
ROTATION_SLICE = slice(BODY_POSITION_DIM, BODY_POSITION_DIM + BODY_ROTATION_DIM)
VELOCITY_SLICE = slice(BODY_POSITION_DIM + BODY_ROTATION_DIM, BODY_CONTINUOUS_DIM)
CONTACT_SLICE = slice(BODY_CONTINUOUS_DIM, BODY_DIM)


def pack_body(
    joint_positions: torch.Tensor,
    joint_rotations: torch.Tensor,
    joint_velocities: torch.Tensor,
    foot_contacts: torch.Tensor,
) -> torch.Tensor:
    """Pack structured body features into physical body259.

    Args:
        joint_positions: Non-root positions ``[..., 21, 3]``.
        joint_rotations: Heading-frame cumulative rotation6d ``[..., 21, 6]``.
        joint_velocities: Current-heading-local velocities ``[..., 22, 3]`` in m/s.
        foot_contacts: Binary contact values ``[..., 4]``.

    Returns:
        Physical body motion ``[..., 259]``.
    """
    prefix = joint_positions.shape[:-2]
    for value, tail, name in (
        (joint_positions, (NUM_JOINTS - 1, 3), "joint_positions"),
        (joint_rotations, (NUM_JOINTS - 1, 6), "joint_rotations"),
        (joint_velocities, (NUM_JOINTS, 3), "joint_velocities"),
        (foot_contacts, (BODY_CONTACT_DIM,), "foot_contacts"),
    ):
        if value.shape[: -len(tail)] != prefix or tuple(value.shape[-len(tail) :]) != tail:
            raise ValueError(f"{name} has incompatible shape {tuple(value.shape)}")
    return torch.cat(
        [
            joint_positions.flatten(-2),
            joint_rotations.flatten(-2),
            joint_velocities.flatten(-2),
            foot_contacts,
        ],
        dim=-1,
    )


def unpack_body(body_motion: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack physical body259 into its four structured feature blocks.

    Args:
        body_motion: Physical body motion ``[..., 259]``.

    Returns:
        Dictionary containing joint positions, rotation6d, velocities and
        contacts with the structured shapes documented at module level.
    """
    if body_motion.shape[-1] != BODY_DIM:
        raise ValueError(f"body_motion must end in {BODY_DIM}")
    return {
        "joint_positions": body_motion[..., POSITION_SLICE].reshape(
            *body_motion.shape[:-1], NUM_JOINTS - 1, 3
        ),
        "joint_rotations": body_motion[..., ROTATION_SLICE].reshape(
            *body_motion.shape[:-1], NUM_JOINTS - 1, 6
        ),
        "joint_velocities": body_motion[..., VELOCITY_SLICE].reshape(
            *body_motion.shape[:-1], NUM_JOINTS, 3
        ),
        "foot_contacts": body_motion[..., CONTACT_SLICE],
    }


def rotation_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Convert body259 continuous 6D rotations to rotation matrices.

    Args:
        rotation: Tensor ending in six rotation features.

    Returns:
        Orthonormal rotation matrices ending in ``[3, 3]``. The two stored
        vectors become the first two matrix columns after Gram-Schmidt.
    """
    if rotation.shape[-1] != 6:
        raise ValueError("rotation must end in six dimensions")
    first = F.normalize(rotation[..., :3], dim=-1)
    second = rotation[..., 3:]
    second = F.normalize(
        second - (first * second).sum(-1, keepdim=True) * first,
        dim=-1,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack([first, second, third], dim=-1)


def matrix_to_rotation(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to the body259 continuous 6D convention.

    Args:
        matrix: Rotation matrices ending in ``[3, 3]``.

    Returns:
        The first two matrix columns concatenated as ``[..., 6]``.
    """
    if tuple(matrix.shape[-2:]) != (3, 3):
        raise ValueError("rotation matrix must end in [3,3]")
    return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)


def project_root_heading(root_motion: torch.Tensor) -> torch.Tensor:
    """Project root5 heading channels onto the unit circle."""

    if root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must end in five root features")
    heading = root_motion[..., 3:5]
    norm = heading.norm(dim=-1, keepdim=True)
    projected = heading / norm.clamp_min(1e-8)
    fallback = torch.zeros_like(projected)
    fallback[..., 0] = 1
    projected = torch.where(norm > 1e-8, projected, fallback)
    return torch.cat([root_motion[..., :3], projected], dim=-1)


def compute_joint_velocities(
    global_positions: torch.Tensor,
    *,
    fps: float = 20.0,
    previous_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute backward global joint velocities and element validity.

    Args:
        global_positions: World joint positions ``[B, F, 22, 3]`` in metres.
        fps: Motion frame rate used to convert displacement to m/s.
        previous_positions: Optional preceding world frame ``[B, 22, 3]``.

    Returns:
        Velocities and boolean validity, both ``[B, F, 22, 3]``. At a cold
        start the first velocity is zero and invalid.
    """
    if global_positions.ndim != 4 or tuple(global_positions.shape[-2:]) != (
        NUM_JOINTS,
        3,
    ):
        raise ValueError("global_positions must be [B,F,22,3]")
    if not global_positions.is_floating_point():
        raise TypeError("global_positions must use a floating dtype")
    if not torch.isfinite(global_positions).all():
        raise ValueError("global_positions must contain only finite values")
    if not float(fps) > 0:
        raise ValueError("fps must be positive")

    batch = global_positions.shape[0]
    cold_start = previous_positions is None
    if cold_start:
        previous = global_positions[:, :1]
    else:
        if tuple(previous_positions.shape) != (batch, NUM_JOINTS, 3):
            raise ValueError("previous_positions must be [B,22,3]")
        previous = previous_positions[:, None].to(global_positions)
    prior = torch.cat([previous, global_positions[:, :-1]], dim=1)
    velocity = (global_positions - prior) * float(fps)
    valid = torch.ones_like(velocity, dtype=torch.bool)
    if cold_start:
        velocity[:, 0] = 0
        valid[:, 0] = False
    return velocity, valid


def build_motion(
    global_positions: torch.Tensor,
    heading_frame_rotations: torch.Tensor,
    root_positions: torch.Tensor,
    root_yaw: torch.Tensor,
    foot_contacts: torch.Tensor,
    *,
    fps: float = 20.0,
    previous_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build physical root5/body259 from world-space skeleton features.

    Args:
        global_positions: World joint positions ``[B, F, 22, 3]``.
        heading_frame_rotations: Root-heading-frame cumulative IK rotation
            matrices ``[B, F, 21, 3, 3]``.
        root_positions: World root translation ``[B, F, 3]``.
        root_yaw: Physical yaw ``[B, F]`` where zero faces +Z.
        foot_contacts: Binary contacts ``[B, F, 4]``.
        fps: Motion frame rate.
        previous_positions: Optional preceding world joints ``[B, 22, 3]``.

    Returns:
        ``(root_motion, body_motion, body_feature_valid_mask)`` with shapes
        ``[B,F,5]``, ``[B,F,259]`` and ``[B,F,259]``.
    """
    if global_positions.ndim != 4 or tuple(global_positions.shape[-2:]) != (
        NUM_JOINTS,
        3,
    ):
        raise ValueError("global_positions must be [B,F,22,3]")
    batch, frames = global_positions.shape[:2]
    if tuple(heading_frame_rotations.shape) != (
        batch,
        frames,
        NUM_JOINTS - 1,
        3,
        3,
    ):
        raise ValueError("heading_frame_rotations must be [B,F,21,3,3]")
    if tuple(root_positions.shape) != (batch, frames, 3):
        raise ValueError("root_positions must be [B,F,3]")
    if tuple(root_yaw.shape) != (batch, frames):
        raise ValueError("root_yaw must be [B,F]")
    if tuple(foot_contacts.shape) != (batch, frames, BODY_CONTACT_DIM):
        raise ValueError("foot_contacts must be [B,F,4]")
    for name, value in (
        ("global_positions", global_positions),
        ("heading_frame_rotations", heading_frame_rotations),
        ("root_positions", root_positions),
        ("root_yaw", root_yaw),
        ("foot_contacts", foot_contacts),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if not float(fps) > 0:
        raise ValueError("fps must be positive")
    if not torch.allclose(
        root_positions,
        global_positions[..., 0, :],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("root_positions must equal global joint zero")
    identity = torch.eye(
        3,
        dtype=heading_frame_rotations.dtype,
        device=heading_frame_rotations.device,
    )
    gram = heading_frame_rotations.transpose(-1, -2) @ heading_frame_rotations
    if not torch.allclose(gram, identity.expand_as(gram), atol=1e-4, rtol=1e-4):
        raise ValueError("heading_frame_rotations must be orthogonal")
    determinant = torch.linalg.det(heading_frame_rotations)
    if not torch.allclose(
        determinant, torch.ones_like(determinant), atol=1e-4, rtol=1e-4
    ):
        raise ValueError("heading_frame_rotations must have determinant one")
    if bool(((foot_contacts < 0) | (foot_contacts > 1)).any()):
        raise ValueError("foot_contacts must be in [0,1]")

    root_motion = torch.cat(
        [
            root_positions,
            torch.cos(root_yaw)[..., None],
            torch.sin(root_yaw)[..., None],
        ],
        dim=-1,
    )

    root_rotation = yaw_to_matrix(root_yaw)
    world_relative = global_positions[..., 1:, :] - root_positions[..., None, :]
    joint_positions = torch.einsum(
        "bfij,bfkj->bfki", root_rotation.transpose(-1, -2), world_relative
    )
    world_velocities, velocity_valid = compute_joint_velocities(
        global_positions,
        fps=fps,
        previous_positions=previous_positions,
    )
    joint_velocities = torch.einsum(
        "bfij,bfkj->bfki",
        root_rotation.transpose(-1, -2),
        world_velocities,
    )
    body_motion = pack_body(
        joint_positions,
        matrix_to_rotation(heading_frame_rotations),
        joint_velocities,
        foot_contacts.to(global_positions),
    )
    feature_valid = torch.ones_like(body_motion, dtype=torch.bool)
    feature_valid[..., VELOCITY_SLICE] = velocity_valid.flatten(-2)
    return project_root_heading(root_motion), body_motion, feature_valid


def recover_root_yaw(root_motion: torch.Tensor) -> torch.Tensor:
    """Recover physical yaw for every explicit root5 frame.

    Args:
        root_motion: Physical root motion ending in ``[F, 5]``.

    Returns:
        Physical yaw ending in ``[F]``. Use ``yaw[..., 0]`` for the initial
        frame heading.
    """
    if root_motion.ndim < 2 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must end in [F,5]")
    if root_motion.shape[-2] < 1:
        raise ValueError("root_motion must contain at least one frame")
    heading = root_motion[..., 3:5]
    if not torch.isfinite(heading).all():
        raise ValueError("root heading must contain only finite values")
    if (heading.square().sum(dim=-1) <= 1e-12).any():
        raise ValueError("root heading must be non-zero")
    return torch.atan2(heading[..., 1], heading[..., 0])


def recover_local_root(
    root_motion: torch.Tensor,
    previous_root_frame: torch.Tensor | None,
    *,
    fps: float = 20.0,
    previous_root_valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover backward/current-heading-local root4 patches.

    Args:
        root_motion: Physical root5 frames ``[B, F, 5]`` with ``F % 4 == 0``.
        previous_root_frame: Optional physical preceding root ``[B, 5]``.
        fps: Motion frame rate used for yaw and planar velocity rates.
        previous_root_valid_mask: Optional boundary validity ``[B]``.

    Returns:
        Local root values and feature validity, both ``[B, T, 4, 4]`` in
        ``[yaw_rate, local_vx, local_vz, root_y]`` order.
    """
    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [B,F,5]")
    require_aligned_frame_count(root_motion.shape[1])
    batch, frames = root_motion.shape[:2]
    tokens = frame_count_to_token_count(frames)
    flat = project_root_heading(root_motion)
    cold_start = previous_root_frame is None
    if cold_start:
        previous = flat[:, :1].clone()
    else:
        if tuple(previous_root_frame.shape) != (batch, ROOT_DIM):
            raise ValueError("previous_root_frame must be [B,5]")
        previous = project_root_heading(previous_root_frame)[:, None].to(flat)
    prior = torch.cat([previous, flat[:, :-1]], dim=1)

    world_displacement = flat[..., [0, 2]] - prior[..., [0, 2]]
    cos_yaw = flat[..., 3]
    sin_yaw = flat[..., 4]
    yaw = torch.atan2(sin_yaw, cos_yaw)
    local_velocity = (
        rotate_vectors_world_to_local(world_displacement, yaw) * float(fps)
    )
    sin_delta = flat[..., 4] * prior[..., 3] - flat[..., 3] * prior[..., 4]
    cos_delta = flat[..., 3] * prior[..., 3] + flat[..., 4] * prior[..., 4]
    yaw_rate = torch.atan2(sin_delta, cos_delta) * float(fps)
    values = torch.cat(
        [yaw_rate[..., None], local_velocity, flat[..., 1:2]], dim=-1
    )
    valid = torch.ones_like(values, dtype=torch.bool)
    if cold_start:
        values[:, 0, :3] = 0
        valid[:, 0, :3] = False
    if previous_root_valid_mask is not None:
        if previous_root_frame is None:
            raise ValueError("previous_root_valid_mask requires previous_root_frame")
        if tuple(previous_root_valid_mask.shape) != (root_motion.shape[0],):
            raise ValueError("previous_root_valid_mask must be [B]")
        cold_start = ~previous_root_valid_mask.bool()
        values[cold_start, 0, :3] = 0
        valid[cold_start, 0, :3] = False
    return (
        values.reshape(batch, tokens, FRAMES_PER_TOKEN, LOCAL_ROOT_DIM),
        valid.reshape(batch, tokens, FRAMES_PER_TOKEN, LOCAL_ROOT_DIM),
    )


def recover_joint_positions(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
) -> torch.Tensor:
    """Recover world-space 22-joint positions from root5 and body motion.

    Args:
        root_motion: Physical root5 ``[..., F, 5]``.
        body_motion: Physical body259 or continuous body255 ``[..., F, D]``.

    Returns:
        World joint positions ``[..., F, 22, 3]``.
    """
    if root_motion.ndim < 2 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must end in [F,5]")
    if body_motion.ndim != root_motion.ndim or body_motion.shape[-1] not in (
        BODY_CONTINUOUS_DIM,
        BODY_DIM,
    ):
        raise ValueError(
            f"body_motion must match root rank and end in "
            f"{BODY_CONTINUOUS_DIM} or {BODY_DIM}"
        )
    if root_motion.shape[:-1] != body_motion.shape[:-1]:
        raise ValueError("root_motion and body_motion must share leading dimensions")
    if not torch.isfinite(root_motion).all() or not torch.isfinite(
        body_motion[..., :BODY_POSITION_DIM]
    ).all():
        raise ValueError("root/body positions must contain only finite values")

    local_positions = body_motion[..., :BODY_POSITION_DIM].reshape(
        *body_motion.shape[:-1], NUM_JOINTS - 1, 3
    )
    root_yaw = recover_root_yaw(root_motion)
    root_rotation = yaw_to_matrix(root_yaw)
    joint_positions = torch.einsum(
        "...fij,...fkj->...fki", root_rotation, local_positions
    ) + root_motion[..., None, :3]
    return torch.cat([root_motion[..., None, :3], joint_positions], dim=-2)


def infer_humanml_skeleton_offsets(
    reference_joint_positions: torch.Tensor,
) -> torch.Tensor:
    """Infer HumanML skeleton offsets from one reference pose per sequence.

    Args:
        reference_joint_positions: World joint positions ending in ``[22,3]``.

    Returns:
        Sequence-specific skeleton offsets with the same leading dimensions and
        a ``[22,3]`` tail. Directions follow HumanML's fixed raw-offset table;
        lengths come from the supplied pose, matching official preprocessing.
    """

    if tuple(reference_joint_positions.shape[-2:]) != (NUM_JOINTS, 3):
        raise ValueError("reference_joint_positions must end in [22,3]")
    if not reference_joint_positions.is_floating_point():
        raise TypeError("reference_joint_positions must use a floating dtype")
    if not torch.isfinite(reference_joint_positions).all():
        raise ValueError("reference_joint_positions must contain only finite values")
    raw = reference_joint_positions.new_tensor(HUMANML22_RAW_OFFSETS)
    offsets = torch.zeros_like(reference_joint_positions)
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        length = (
            reference_joint_positions[..., joint, :]
            - reference_joint_positions[..., parent, :]
        ).norm(dim=-1, keepdim=True)
        offsets[..., joint, :] = length * raw[joint]
    return offsets


def forward_kinematics_heading_frame(
    root_motion: torch.Tensor,
    heading_frame_rotations: torch.Tensor,
    skeleton_offsets: torch.Tensor,
) -> torch.Tensor:
    """Run HumanML-gauge FK from Body259 cumulative rotations.

    ``heading_frame_rotations`` are the cumulative ``B_j`` matrices stored by
    Body259, not parent-local rotations.  World cumulative matrices are
    ``A_j = R B_j`` and rotate each joint's HumanML rest offset directly.
    """

    if root_motion.ndim < 2 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must end in [F,5]")
    expected_rotation_shape = (
        *root_motion.shape[:-1],
        NUM_JOINTS - 1,
        3,
        3,
    )
    if tuple(heading_frame_rotations.shape) != expected_rotation_shape:
        raise ValueError(
            "heading_frame_rotations must match root frames and end in [21,3,3]"
        )
    expected_offset_shape = (*root_motion.shape[:-2], NUM_JOINTS, 3)
    if tuple(skeleton_offsets.shape) not in (
        (NUM_JOINTS, 3),
        expected_offset_shape,
    ):
        raise ValueError("skeleton_offsets must be [22,3] or batch-shaped [B,22,3]")
    if not torch.isfinite(heading_frame_rotations).all() or not torch.isfinite(
        skeleton_offsets
    ).all():
        raise ValueError("FK rotations and offsets must contain only finite values")

    root_rotation = yaw_to_matrix(recover_root_yaw(root_motion))
    world_rotations = root_rotation[..., None, :, :] @ heading_frame_rotations
    positions = [root_motion[..., :3]]
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        offset = skeleton_offsets[..., joint, :]
        while offset.ndim < world_rotations[..., joint - 1, :, :].ndim - 1:
            offset = offset.unsqueeze(-2)
        rotated_offset = torch.einsum(
            "...ij,...j->...i", world_rotations[..., joint - 1, :, :], offset
        )
        positions.append(positions[parent] + rotated_offset)
    return torch.stack(positions, dim=-2)


def _yaw_rotation(
    angle: torch.Tensor,
    reference: torch.Tensor,
    batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    angle = torch.as_tensor(angle, device=reference.device, dtype=reference.dtype)
    if angle.ndim == 0:
        angle = angle.expand(batch)
    if tuple(angle.shape) != (batch,):
        raise ValueError("angle must be scalar or [B]")
    return angle, yaw_to_matrix(angle)


def rotate_root_yaw(
    root_motion: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    """Apply one global yaw offset to each root5 sequence.

    Args:
        root_motion: Physical root5 ``[B, F, 5]``.
        angle: Scalar yaw offset or one value per batch item ``[B]``.

    Returns:
        Rotated physical root5 with unchanged shape.
    """
    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [B,F,5]")
    angle, matrix = _yaw_rotation(angle, root_motion, root_motion.shape[0])
    root = root_motion.clone()
    root[..., :3] = torch.einsum("bij,bfj->bfi", matrix, root[..., :3])
    heading = recover_root_yaw(root_motion) + angle[:, None]
    root[..., 3], root[..., 4] = torch.cos(heading), torch.sin(heading)
    return root


def rotate_motion_yaw(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply global yaw to root5 while leaving invariant body259 unchanged.

    Args:
        root_motion: Physical root5 ``[B, F, 5]``.
        body_motion: Physical body259 ``[B, F, 259]``.
        angle: Scalar yaw offset or one value per batch item ``[B]``.

    Returns:
        Rotated root and the unchanged body tensor.
    """
    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be [B,F,5]")
    if body_motion.ndim != 3 or body_motion.shape[-1] != BODY_DIM:
        raise ValueError(f"body_motion must be [B,F,{BODY_DIM}]")
    if root_motion.shape[:2] != body_motion.shape[:2]:
        raise ValueError("root_motion and body_motion must share [B,F]")
    angle, _ = _yaw_rotation(angle, root_motion, root_motion.shape[0])
    root = rotate_root_yaw(root_motion, angle)
    return root, body_motion.clone()


__all__ = [
    "BODY_CONTACT_DIM",
    "BODY_CONTINUOUS_DIM",
    "BODY_DIM",
    "BODY_POSITION_DIM",
    "BODY_ROTATION_DIM",
    "BODY_VELOCITY_DIM",
    "CONTACT_SLICE",
    "FOOT_JOINT_INDICES",
    "HUMANML22_PARENTS",
    "HUMANML22_RAW_OFFSETS",
    "LEFT_ANKLE_INDEX",
    "LEFT_TOE_INDEX",
    "LOCAL_ROOT_DIM",
    "NUM_JOINTS",
    "POSITION_SLICE",
    "ROOT_DIM",
    "RIGHT_ANKLE_INDEX",
    "RIGHT_TOE_INDEX",
    "ROTATION_SLICE",
    "VELOCITY_SLICE",
    "build_motion",
    "compute_joint_velocities",
    "forward_kinematics_heading_frame",
    "infer_humanml_skeleton_offsets",
    "matrix_to_rotation",
    "pack_body",
    "project_root_heading",
    "recover_joint_positions",
    "recover_local_root",
    "recover_root_yaw",
    "rotate_motion_yaw",
    "rotate_root_yaw",
    "rotation_to_matrix",
    "unpack_body",
]
