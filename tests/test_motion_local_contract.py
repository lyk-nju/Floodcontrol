import math

import pytest
import torch

from utils.coordinate_transform import yaw_to_matrix
from utils.motion_process import (
    BODY_DIM,
    CONTACT_SLICE,
    HUMANML22_PARENTS,
    HUMANML22_RAW_OFFSETS,
    NUM_JOINTS,
    ROTATION_SLICE,
    VELOCITY_SLICE,
    build_motion,
    forward_kinematics_heading_frame,
    infer_humanml_skeleton_offsets,
    recover_joint_positions,
    rotation_to_matrix,
    unpack_body,
)


def _local_motion(frames: int = 8):
    root_yaw = torch.linspace(-0.4, 0.7, frames)[None]
    root_rotation = yaw_to_matrix(root_yaw)
    root_positions = torch.zeros(1, frames, 3)
    root_positions[0, :, 0] = torch.linspace(-0.2, 0.8, frames)
    root_positions[0, :, 1] = torch.linspace(0.9, 1.1, frames)
    root_positions[0, :, 2] = torch.linspace(0.4, -0.3, frames)

    generator = torch.Generator().manual_seed(123)
    local_positions = torch.randn(
        1, frames, NUM_JOINTS - 1, 3, generator=generator
    ) * 0.1
    world_non_root = torch.einsum(
        "bfij,bfkj->bfki", root_rotation, local_positions
    ) + root_positions[..., None, :]
    world_positions = torch.cat(
        [root_positions[..., None, :], world_non_root], dim=-2
    )

    rotation6d = torch.randn(
        1, frames, NUM_JOINTS - 1, 6, generator=generator
    )
    heading_rotations = rotation_to_matrix(rotation6d)
    contacts = torch.randint(
        0, 2, (1, frames, 4), generator=generator
    ).float()
    return (
        world_positions,
        heading_rotations,
        root_positions,
        root_yaw,
        contacts,
    )


def test_body259_contract_dimensions_and_world_position_inverse():
    world, rotations, root_xyz, root_yaw, contacts = _local_motion()
    root, body, valid = build_motion(
        world, rotations, root_xyz, root_yaw, contacts, fps=20.0
    )
    assert root.shape == (1, 8, 5)
    assert body.shape == valid.shape == (1, 8, BODY_DIM)
    assert body.dtype == torch.float32
    assert valid.dtype == torch.bool
    assert torch.isfinite(root).all() and torch.isfinite(body).all()
    assert torch.allclose(root[..., 3:5].norm(dim=-1), torch.ones(1, 8))
    assert torch.equal(body[..., CONTACT_SLICE], contacts)
    assert torch.allclose(recover_joint_positions(root, body), world, atol=1e-6)


def test_body259_is_invariant_to_world_xyz_translation():
    world, rotations, root_xyz, root_yaw, contacts = _local_motion()
    world = world.double()
    rotations = rotations.double()
    root_xyz = root_xyz.double()
    root_yaw = root_yaw.double()
    contacts = contacts.double()
    root, body, valid = build_motion(
        world, rotations, root_xyz, root_yaw, contacts, fps=20.0
    )
    translation = torch.tensor([4.0, -2.5, 7.0], dtype=torch.float64)
    translated_root, translated_body, translated_valid = build_motion(
        world + translation,
        rotations,
        root_xyz + translation,
        root_yaw,
        contacts,
        fps=20.0,
    )
    assert torch.allclose(translated_body, body, atol=1e-6)
    assert torch.equal(translated_valid, valid)
    assert torch.allclose(translated_root[..., :3], root[..., :3] + translation)


@pytest.mark.parametrize("degrees", [0.0, 45.0, 90.0, 180.0, -73.0])
def test_body259_is_invariant_to_shared_world_yaw(degrees: float):
    world, rotations, root_xyz, root_yaw, contacts = _local_motion()
    root, body, valid = build_motion(
        world, rotations, root_xyz, root_yaw, contacts, fps=20.0
    )
    delta = torch.tensor(math.radians(degrees))
    global_rotation = yaw_to_matrix(delta)
    rotated_world = torch.einsum("ij,bfkj->bfki", global_rotation, world)
    rotated_root_xyz = torch.einsum("ij,bfj->bfi", global_rotation, root_xyz)
    rotated_root, rotated_body, rotated_valid = build_motion(
        rotated_world,
        rotations,
        rotated_root_xyz,
        root_yaw + delta,
        contacts,
        fps=20.0,
    )
    assert torch.allclose(rotated_body, body, atol=1e-5, rtol=1e-5)
    assert torch.equal(rotated_valid, valid)
    assert torch.allclose(
        recover_joint_positions(rotated_root, rotated_body),
        rotated_world,
        atol=1e-5,
        rtol=1e-5,
    )


def test_body259_velocity_uses_current_heading_backward_difference():
    world, rotations, root_xyz, root_yaw, contacts = _local_motion()
    _, body, valid = build_motion(
        world, rotations, root_xyz, root_yaw, contacts, fps=20.0
    )
    parts = unpack_body(body)
    expected_world = torch.zeros_like(world)
    expected_world[:, 1:] = (world[:, 1:] - world[:, :-1]) * 20.0
    expected_local = torch.einsum(
        "bfij,bfkj->bfki",
        yaw_to_matrix(root_yaw).transpose(-1, -2),
        expected_world,
    )
    assert torch.equal(parts["joint_velocities"][:, 0], torch.zeros_like(world[:, 0]))
    assert torch.allclose(parts["joint_velocities"], expected_local, atol=1e-6)
    assert not valid[:, 0, VELOCITY_SLICE].any()
    assert valid[:, 1:, VELOCITY_SLICE].all()


def test_heading_frame_rotation6d_is_orthogonal_and_world_relation_is_a_eq_rb():
    _, rotations, root_xyz, root_yaw, contacts = _local_motion()
    root_rotation = yaw_to_matrix(root_yaw)
    local_positions = torch.zeros(1, root_xyz.shape[1], NUM_JOINTS - 1, 3)
    world_non_root = torch.einsum(
        "bfij,bfkj->bfki", root_rotation, local_positions
    ) + root_xyz[..., None, :]
    world = torch.cat([root_xyz[..., None, :], world_non_root], dim=-2)
    _, body, _ = build_motion(
        world, rotations, root_xyz, root_yaw, contacts, fps=20.0
    )
    rebuilt_b = rotation_to_matrix(
        body[..., ROTATION_SLICE].reshape(
            1, root_xyz.shape[1], NUM_JOINTS - 1, 6
        )
    )
    identity = rebuilt_b.transpose(-1, -2) @ rebuilt_b
    determinant = torch.linalg.det(rebuilt_b)
    expected_world_rotation = root_rotation[..., None, :, :] @ rotations
    rebuilt_world_rotation = root_rotation[..., None, :, :] @ rebuilt_b
    assert torch.allclose(
        identity, torch.eye(3).expand_as(identity), atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(determinant, torch.ones_like(determinant), atol=1e-5)
    assert torch.allclose(
        rebuilt_world_rotation, expected_world_rotation, atol=1e-5, rtol=1e-5
    )


def test_humanml_fk_uses_reference_scale_and_cumulative_child_rotation():
    frames = 4
    root = torch.zeros(frames, 5)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    reference = torch.zeros(NUM_JOINTS, 3)
    # Build a topologically valid reference pose with non-unit bone lengths.
    for joint in range(1, NUM_JOINTS):
        parent = HUMANML22_PARENTS[joint]
        direction = torch.tensor(HUMANML22_RAW_OFFSETS[joint])
        reference[joint] = reference[parent] + direction * (0.1 + joint * 0.01)
    offsets = infer_humanml_skeleton_offsets(reference)
    identity = torch.eye(3).expand(frames, NUM_JOINTS - 1, 3, 3).clone()
    joints = forward_kinematics_heading_frame(root, identity, offsets)
    expected = reference[None].expand(frames, -1, -1).clone()
    expected[..., 1] += 1.0
    assert torch.allclose(joints, expected, atol=1e-6)
