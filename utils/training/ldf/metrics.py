"""Observation-only heading metrics for hybrid LDF predictions."""

from __future__ import annotations

import torch

from utils.motion_process import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    ROOT_DIM,
    project_root_heading,
    recover_joint_positions,
)


# HumanML 22-joint order used by the motion representation.  Body facing is
# estimated from the same hip/shoulder geometry used by HumanML preprocessing,
# rather than from the global root rotation stored in body265.  The latter has
# a dataset-specific IK gauge and is therefore not directly comparable to the
# physical root5 heading.
_RIGHT_HIP = 2
_LEFT_HIP = 1
_RIGHT_SHOULDER = 17
_LEFT_SHOULDER = 16
_LEFT_ANKLE = 7
_RIGHT_ANKLE = 8
_LEFT_TOE = 10
_RIGHT_TOE = 11


def _validate_inputs(
    predicted_root: torch.Tensor,
    target_root: torch.Tensor,
    predicted_body: torch.Tensor,
    frame_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor,
) -> None:
    if predicted_root.ndim != 3 or predicted_root.shape[-1] != ROOT_DIM:
        raise ValueError("predicted_root must be physical [B,F,5]")
    if tuple(target_root.shape) != tuple(predicted_root.shape):
        raise ValueError("target_root must match predicted_root")
    if predicted_body.ndim != 3 or predicted_body.shape[-1] not in (
        BODY_CONTINUOUS_DIM,
        BODY_DIM,
    ):
        raise ValueError("predicted_body must be physical [B,F,261 or 265]")
    if tuple(predicted_body.shape[:2]) != tuple(predicted_root.shape[:2]):
        raise ValueError("predicted body and root must share [B,F]")
    if tuple(frame_mask.shape) != tuple(predicted_root.shape[:2]):
        raise ValueError("frame_mask must match root [B,F]")
    if frame_mask.dtype != torch.bool:
        raise TypeError("frame_mask must be bool [B,F]")
    if tuple(frame_valid_mask.shape) != tuple(predicted_root.shape[:2]):
        raise ValueError("frame_valid_mask must match root [B,F]")
    if frame_valid_mask.dtype != torch.bool:
        raise TypeError("frame_valid_mask must be bool [B,F]")
    if not (
        predicted_root.device
        == target_root.device
        == predicted_body.device
        == frame_mask.device
        == frame_valid_mask.device
    ):
        raise ValueError("heading metric inputs must share one device")


def _mean_angle_degrees(
    first_direction: torch.Tensor,
    second_direction: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return a stable masked unsigned angle in degrees."""

    first = torch.nn.functional.normalize(first_direction.float(), dim=-1)
    second = torch.nn.functional.normalize(second_direction.float(), dim=-1)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    dot = (first * second).sum(dim=-1)
    angle = torch.rad2deg(torch.atan2(cross.abs(), dot))
    weight = mask.to(dtype=angle.dtype)
    return (angle * weight).sum() / weight.sum().clamp_min(1.0)


def _root_forward_direction(root_motion: torch.Tensor) -> torch.Tensor:
    root = project_root_heading(root_motion)
    # root5 stores [cos(yaw), sin(yaw)], while a physical forward vector in
    # XZ order is [sin(yaw), cos(yaw)].
    return root[..., [4, 3]]


def _body_forward_direction(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    joints = recover_joint_positions(root_motion, body_motion)
    across = (
        joints[..., _RIGHT_HIP, :] - joints[..., _LEFT_HIP, :]
        + joints[..., _RIGHT_SHOULDER, :]
        - joints[..., _LEFT_SHOULDER, :]
    )
    across_xz = across[..., [0, 2]].float()
    # HumanML defines forward = cross(world_up, right-minus-left).  In XZ
    # coordinates this is [across_z, -across_x].
    forward = torch.stack([across_xz[..., 1], -across_xz[..., 0]], dim=-1)
    valid = torch.isfinite(forward).all(dim=-1) & (forward.norm(dim=-1) > 1e-6)
    return forward, valid


def _foot_forward_directions(
    root_motion: torch.Tensor,
    body_motion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct-position ankle-to-toe directions for both feet."""

    joints = recover_joint_positions(root_motion, body_motion)
    directions = torch.stack(
        [
            joints[..., _LEFT_TOE, [0, 2]]
            - joints[..., _LEFT_ANKLE, [0, 2]],
            joints[..., _RIGHT_TOE, [0, 2]]
            - joints[..., _RIGHT_ANKLE, [0, 2]],
        ],
        dim=-2,
    ).float()
    valid = torch.isfinite(directions).all(dim=-1) & (
        directions.norm(dim=-1) > 1e-6
    )
    return directions, valid


def _reverse_ratio(
    first_direction: torch.Tensor,
    second_direction: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    first = torch.nn.functional.normalize(first_direction.float(), dim=-1)
    second = torch.nn.functional.normalize(second_direction.float(), dim=-1)
    reverse = ((first * second).sum(dim=-1) < 0.0).float()
    weight = mask.to(dtype=reverse.dtype)
    return (reverse * weight).sum() / weight.sum().clamp_min(1.0)


def _trajectory_forward_direction(
    target_root: torch.Tensor,
    frame_valid_mask: torch.Tensor,
    *,
    fps: float,
    minimum_speed_mps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive causal GT path tangents from backward XZ displacement."""

    velocity = target_root.new_zeros(*target_root.shape[:2], 2).float()
    velocity[:, 1:] = (
        target_root[:, 1:, [0, 2]].float()
        - target_root[:, :-1, [0, 2]].float()
    ) * float(fps)
    valid = torch.zeros_like(frame_valid_mask)
    valid[:, 1:] = frame_valid_mask[:, 1:] & frame_valid_mask[:, :-1]
    valid &= torch.isfinite(velocity).all(dim=-1)
    valid &= velocity.norm(dim=-1) >= float(minimum_speed_mps)
    return velocity, valid


def compute_heading_metrics(
    *,
    predicted_root: torch.Tensor,
    target_root: torch.Tensor,
    predicted_body: torch.Tensor,
    frame_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor | None = None,
    fps: float = 20.0,
    minimum_trajectory_speed_mps: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Measure Root Stage heading accuracy and root/body facing consistency.

    All motion inputs are physical frame-space tensors.  GT trajectory heading
    uses causal backward XZ displacement and excludes near-stationary frames.
    Body heading follows decoded joint geometry, so it measures the facing
    visible in rendered motion and remains independent of body rotation gauge.
    """

    if float(fps) <= 0.0:
        raise ValueError("fps must be positive")
    if float(minimum_trajectory_speed_mps) < 0.0:
        raise ValueError("minimum_trajectory_speed_mps must be non-negative")
    if frame_valid_mask is None:
        frame_valid_mask = frame_mask
    _validate_inputs(
        predicted_root,
        target_root,
        predicted_body,
        frame_mask,
        frame_valid_mask,
    )
    predicted_direction = _root_forward_direction(predicted_root)
    target_direction = _root_forward_direction(target_root)
    body_direction, body_valid = _body_forward_direction(
        predicted_root, predicted_body
    )
    foot_direction, foot_valid = _foot_forward_directions(
        predicted_root, predicted_body
    )
    trajectory_direction, trajectory_valid = _trajectory_forward_direction(
        target_root,
        frame_valid_mask,
        fps=float(fps),
        minimum_speed_mps=float(minimum_trajectory_speed_mps),
    )
    return {
        "root_gt_heading_angle_deg": _mean_angle_degrees(
            predicted_direction,
            target_direction,
            frame_mask,
        ),
        "root_gt_trajectory_heading_angle_deg": _mean_angle_degrees(
            predicted_direction,
            trajectory_direction,
            frame_mask & trajectory_valid,
        ),
        "root_body_heading_angle_deg": _mean_angle_degrees(
            predicted_direction,
            body_direction,
            frame_mask & body_valid,
        ),
        "feet_root_reverse_ratio": _reverse_ratio(
            foot_direction,
            predicted_direction[..., None, :].expand_as(foot_direction),
            frame_mask[..., None] & foot_valid,
        ),
    }


__all__ = ["compute_heading_metrics"]
