import math

import pytest
import torch

from metrics.humanml import convert_root5_body259_to_humanml263
from tools.convert_motion_263_to_259 import (
    HUMANML_CONTACT_SLICE,
    HUMANML_ROTATION_SLICE,
    convert_motion_263_to_259,
    recover_joint_positions_263,
    recover_joint_rotations_263,
    recover_root_263,
)
from utils.math.quaternion import qrot
from utils.coordinate_transform import yaw_to_matrix
from utils.motion_process import (
    ROTATION_SLICE,
    recover_root_yaw,
    rotate_motion_yaw,
    rotation_to_matrix,
)


def _coherent_humanml_motion(frames: int = 13) -> torch.Tensor:
    motion = torch.zeros(frames, 263)
    motion[:-1, 0] = torch.linspace(-0.02, 0.03, frames - 1)
    motion[:-1, 1] = torch.linspace(0.005, 0.02, frames - 1)
    motion[:-1, 2] = torch.linspace(-0.01, 0.015, frames - 1)
    motion[:, 3] = 1.0
    positions = torch.randn(frames, 21, 3) * 0.05
    positions[..., 1] += 1.0
    motion[:, 4:67] = positions.flatten(-2)
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    motion[:, HUMANML_ROTATION_SLICE] = identity.repeat(21)
    motion[:, HUMANML_CONTACT_SLICE] = torch.randint(0, 2, (frames, 4)).float()

    heading, root = recover_root_263(motion)
    world = recover_joint_positions_263(
        motion, canonical_heading=heading, root_positions=root
    )
    velocity = qrot(
        heading[:-1, None, :].expand(frames - 1, 22, 4),
        world[1:] - world[:-1],
    )
    motion[:-1, 193:259] = velocity.flatten(-2)
    return motion


def test_humanml_adapter_exactly_roundtrips_observable_rows():
    source = _coherent_humanml_motion()
    root, body, _ = convert_motion_263_to_259(source)
    rebuilt = convert_root5_body259_to_humanml263(root, body)
    assert rebuilt.shape == (source.shape[0] - 1, 263)
    assert torch.allclose(rebuilt, source[:-1], atol=2e-5, rtol=1e-5)


def test_humanml_rotation_contract_distinguishes_canonical_c_and_physical_r():
    source = _coherent_humanml_motion()
    canonical, _ = recover_root_263(source)
    cumulative_a = recover_joint_rotations_263(
        source, canonical_heading=canonical
    )
    root, body, _ = convert_motion_263_to_259(source)
    physical_r = yaw_to_matrix(recover_root_yaw(root))
    stored_b = rotation_to_matrix(
        body[:, ROTATION_SLICE].reshape(source.shape[0], 21, 6)
    )
    assert torch.allclose(
        physical_r[..., None, :, :] @ stored_b,
        cumulative_a[..., 1:, :, :],
        atol=2e-5,
        rtol=1e-5,
    )


def test_humanml_adapter_supports_batch_and_explicit_approximate_tail():
    source = _coherent_humanml_motion()
    root, body, _ = convert_motion_263_to_259(source)
    rebuilt = convert_root5_body259_to_humanml263(
        root[None].repeat(2, 1, 1),
        body[None].repeat(2, 1, 1),
        tail="approximate",
    )
    assert rebuilt.shape == (2, source.shape[0], 263)
    assert torch.isfinite(rebuilt).all()


@pytest.mark.parametrize("degrees", [0.0, 45.0, 90.0, 180.0, -67.0])
def test_humanml_adapter_removes_arbitrary_global_yaw(degrees: float):
    source = _coherent_humanml_motion()
    root, body, _ = convert_motion_263_to_259(source)
    rotated_root, rotated_body = rotate_motion_yaw(
        root[None], body[None], torch.tensor([math.radians(degrees)])
    )
    original = convert_root5_body259_to_humanml263(root, body)
    rotated = convert_root5_body259_to_humanml263(
        rotated_root[0], rotated_body[0]
    )
    assert torch.allclose(original, rotated, atol=2e-5, rtol=1e-5)
