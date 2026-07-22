from __future__ import annotations

import torch

from utils.motion_process import BODY_CONTINUOUS_DIM
from utils.training.ldf.metrics import compute_heading_metrics


def _root(yaw_degrees: list[float]) -> torch.Tensor:
    yaw = torch.deg2rad(torch.tensor(yaw_degrees))
    root = torch.zeros(1, len(yaw_degrees), 5)
    root[..., 3] = yaw.cos()
    root[..., 4] = yaw.sin()
    return root


def _body_facing(direction_xz: list[tuple[float, float]]) -> torch.Tensor:
    body = torch.zeros(1, len(direction_xz), BODY_CONTINUOUS_DIM)
    positions = body[..., :63].reshape(1, len(direction_xz), 21, 3)
    for frame, (forward_x, forward_z) in enumerate(direction_xz):
        # forward = [across_z, -across_x]
        across_x = -float(forward_z)
        across_z = float(forward_x)
        right = torch.tensor([across_x, 0.0, across_z])
        left = -right
        positions[0, frame, 2 - 1] = right
        positions[0, frame, 1 - 1] = left
        positions[0, frame, 17 - 1] = right
        positions[0, frame, 16 - 1] = left
    return body


def _set_feet_facing(
    body: torch.Tensor,
    directions_xz: list[tuple[float, float]],
) -> torch.Tensor:
    positions = body[..., :63].reshape(body.shape[0], body.shape[1], 21, 3)
    for frame, (direction_x, direction_z) in enumerate(directions_xz):
        offset = torch.tensor([direction_x, 0.0, direction_z])
        positions[0, frame, 10 - 1] = positions[0, frame, 7 - 1] + offset
        positions[0, frame, 11 - 1] = positions[0, frame, 8 - 1] + offset
    return body


def test_heading_metrics_measure_root_target_and_rendered_body_angles():
    predicted_root = _root([0.0, 90.0, 180.0])
    target_root = _root([0.0, 0.0, 0.0])
    predicted_body = _body_facing(
        [
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ]
    )
    predicted_body = _set_feet_facing(
        predicted_body,
        [(0.0, 1.0), (0.0, 1.0), (0.0, -1.0)],
    )
    target_body = _body_facing([(0.0, 1.0)] * 3)
    target_body = _set_feet_facing(target_body, [(0.0, 1.0)] * 3)
    metrics = compute_heading_metrics(
        predicted_root=predicted_root,
        target_root=target_root,
        predicted_body=predicted_body,
        target_body=target_body,
        frame_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    assert torch.allclose(
        metrics["root_gt_root_heading_angle_deg"], torch.tensor(90.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_trajectory_heading_angle_deg"],
        torch.tensor(0.0),
        atol=1e-5,
    )
    assert torch.allclose(
        metrics["root_body_heading_angle_deg"], torch.tensor(60.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_feet_heading_angle_deg"], torch.tensor(60.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["body_gt_body_heading_angle_deg"], torch.tensor(30.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["feet_gt_feet_heading_angle_deg"], torch.tensor(30.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["gt_root_body_heading_angle_deg"], torch.tensor(0.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["gt_root_feet_heading_angle_deg"], torch.tensor(0.0), atol=1e-5
    )


def test_heading_metrics_respect_frame_mask_and_ignore_degenerate_body_axis():
    predicted_root = _root([0.0, 90.0, 180.0])
    target_root = _root([0.0, 0.0, 0.0])
    predicted_body = _body_facing([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    predicted_body[:, 1, :63] = 0.0
    target_body = _body_facing([(0.0, 1.0)] * 3)
    metrics = compute_heading_metrics(
        predicted_root=predicted_root,
        target_root=target_root,
        predicted_body=predicted_body,
        target_body=target_body,
        frame_mask=torch.tensor([[True, True, False]]),
    )

    assert torch.allclose(
        metrics["root_gt_root_heading_angle_deg"], torch.tensor(45.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_trajectory_heading_angle_deg"],
        torch.tensor(0.0),
        atol=1e-5,
    )
    assert torch.allclose(
        metrics["root_body_heading_angle_deg"], torch.tensor(0.0), atol=1e-5
    )


def test_trajectory_heading_uses_backward_gt_xz_and_filters_stationary_frames():
    predicted_root = _root([0.0, 0.0, 0.0, 0.0])
    target_root = _root([0.0, 0.0, 0.0, 0.0])
    target_root[0, :, 0] = torch.tensor([0.0, 0.1, 0.1, 0.2])
    predicted_body = _body_facing([(0.0, 1.0)] * 4)
    target_body = _body_facing([(0.0, 1.0)] * 4)
    metrics = compute_heading_metrics(
        predicted_root=predicted_root,
        target_root=target_root,
        predicted_body=predicted_body,
        target_body=target_body,
        frame_mask=torch.ones(1, 4, dtype=torch.bool),
        frame_valid_mask=torch.ones(1, 4, dtype=torch.bool),
        fps=20.0,
    )

    # Frames 1 and 3 move toward +X while frame 0 has no predecessor and frame
    # 2 is stationary. A +Z root heading is therefore 90 degrees away.
    assert torch.allclose(
        metrics["root_trajectory_heading_angle_deg"],
        torch.tensor(90.0),
        atol=1e-5,
    )


def test_root_feet_reverse_ratio_uses_ankle_to_toe_geometry():
    root = _root([0.0, 0.0])
    body = _body_facing([(0.0, 1.0), (0.0, 1.0)])
    body = _set_feet_facing(body, [(0.0, 1.0), (0.0, -1.0)])

    metrics = compute_heading_metrics(
        predicted_root=root,
        target_root=root,
        predicted_body=body,
        target_body=body,
        frame_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert torch.allclose(
        metrics["root_feet_reverse_ratio"], torch.tensor(0.5), atol=1e-6
    )


def test_heading_metrics_compare_generated_body_and_feet_to_gt_geometry():
    root = _root([0.0])
    predicted_body = _body_facing([(1.0, 0.0)])
    predicted_body = _set_feet_facing(predicted_body, [(0.0, -1.0)])
    target_body = _body_facing([(0.0, 1.0)])
    target_body = _set_feet_facing(target_body, [(0.0, 1.0)])

    metrics = compute_heading_metrics(
        predicted_root=root,
        target_root=root,
        predicted_body=predicted_body,
        target_body=target_body,
        frame_mask=torch.ones(1, 1, dtype=torch.bool),
    )

    assert torch.allclose(
        metrics["body_gt_body_heading_angle_deg"], torch.tensor(90.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["feet_gt_feet_heading_angle_deg"], torch.tensor(180.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_body_heading_angle_deg"], torch.tensor(90.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_feet_heading_angle_deg"], torch.tensor(180.0), atol=1e-5
    )
    assert torch.allclose(
        metrics["root_feet_reverse_ratio"], torch.tensor(1.0), atol=1e-6
    )
