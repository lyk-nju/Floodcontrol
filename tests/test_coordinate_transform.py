import math

import torch

from utils.coordinate_transform import (
    heading_to_direction,
    matrix_to_yaw,
    rotate_vectors_local_to_world,
    rotate_vectors_world_to_local,
    transform_points_local_to_world,
    transform_points_world_to_local,
    wrap_angle,
    yaw_to_matrix,
)


def test_yaw_matrix_and_heading_direction_share_one_convention():
    yaw = torch.tensor([-math.pi / 2, 0.0, math.pi / 2])
    matrix = yaw_to_matrix(yaw)
    forward = torch.tensor([0.0, 0.0, 1.0]).expand(3, -1)
    rotated = torch.einsum("bij,bj->bi", matrix, forward)
    assert torch.allclose(rotated[:, [0, 2]], heading_to_direction(yaw), atol=1e-6)
    assert torch.allclose(
        wrap_angle(matrix_to_yaw(matrix) - yaw), torch.zeros(3), atol=1e-6
    )


def test_point_transforms_roundtrip_with_prefix_anchor_broadcasting():
    points_world = torch.tensor(
        [
            [[2.0, 4.0], [3.0, 5.0]],
            [[-1.0, 2.0], [0.0, 4.0]],
        ]
    )
    origin = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])
    yaw = torch.tensor([math.pi / 2, -math.pi / 2])
    local = transform_points_world_to_local(points_world, origin, yaw)
    rebuilt = transform_points_local_to_world(local, origin, yaw)
    assert torch.allclose(rebuilt, points_world, atol=1e-6)


def test_vector_transforms_roundtrip_without_translation():
    vectors_world = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
    yaw = torch.tensor([math.pi / 3])
    local = rotate_vectors_world_to_local(vectors_world, yaw)
    rebuilt = rotate_vectors_local_to_world(local, yaw)
    assert torch.allclose(rebuilt, vectors_world, atol=1e-6)

    shifted_origin = torch.tensor([[100.0, -50.0]])
    transformed_point = transform_points_world_to_local(
        vectors_world, shifted_origin, yaw
    )
    assert not torch.allclose(transformed_point, local)
