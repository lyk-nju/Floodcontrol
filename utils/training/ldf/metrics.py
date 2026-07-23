"""Observation-only heading metrics for hybrid LDF predictions."""

from __future__ import annotations

import torch

from utils.motion_process import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    LEFT_ANKLE_INDEX,
    LEFT_TOE_INDEX,
    ROOT_DIM,
    RIGHT_ANKLE_INDEX,
    RIGHT_TOE_INDEX,
    project_root_heading,
    recover_joint_positions,
)


# HumanML 22-joint order used by the motion representation.  Body facing is
# estimated from the same hip/shoulder geometry used by HumanML preprocessing,
# rather than from the IK-derived rotation block stored in body259. The latter
# is retained for HumanML reconstruction and is not a physical-facing label.
_RIGHT_HIP = 2
_LEFT_HIP = 1
_RIGHT_SHOULDER = 17
_LEFT_SHOULDER = 16


def _validate_inputs(
    predicted_root: torch.Tensor,
    target_root: torch.Tensor,
    predicted_body: torch.Tensor,
    target_body: torch.Tensor,
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
        raise ValueError(
            "predicted_body must be physical [B,F,"
            f"{BODY_CONTINUOUS_DIM} or {BODY_DIM}]"
        )
    if tuple(predicted_body.shape[:2]) != tuple(predicted_root.shape[:2]):
        raise ValueError("predicted body and root must share [B,F]")
    if target_body.ndim != 3 or target_body.shape[-1] not in (
        BODY_CONTINUOUS_DIM,
        BODY_DIM,
    ):
        raise ValueError(
            "target_body must be physical [B,F,"
            f"{BODY_CONTINUOUS_DIM} or {BODY_DIM}]"
        )
    if tuple(target_body.shape[:2]) != tuple(predicted_root.shape[:2]):
        raise ValueError("target body and root must share [B,F]")
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
        == target_body.device
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


def _unsigned_angle_degrees(
    first_direction: torch.Tensor,
    second_direction: torch.Tensor,
) -> torch.Tensor:
    first = torch.nn.functional.normalize(first_direction.float(), dim=-1)
    second = torch.nn.functional.normalize(second_direction.float(), dim=-1)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    dot = (first * second).sum(dim=-1)
    return torch.rad2deg(torch.atan2(cross.abs(), dot))


def _signed_angle(
    first_direction: torch.Tensor,
    second_direction: torch.Tensor,
) -> torch.Tensor:
    first = torch.nn.functional.normalize(first_direction.float(), dim=-1)
    second = torch.nn.functional.normalize(second_direction.float(), dim=-1)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    dot = (first * second).sum(dim=-1)
    return torch.atan2(cross, dot)


def _masked_scalar(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    selected = value[mask]
    if selected.numel() == 0:
        return value.new_tensor(float("nan"))
    if reduction == "mean":
        return selected.mean()
    if reduction == "max":
        return selected.max()
    if reduction == "p95":
        return torch.quantile(selected.float(), 0.95)
    raise ValueError(f"unsupported masked reduction {reduction!r}")


def _relative_heading_error_degrees(
    predicted_reference: torch.Tensor,
    predicted_direction: torch.Tensor,
    target_reference: torch.Tensor,
    target_direction: torch.Tensor,
) -> torch.Tensor:
    predicted_relative = _signed_angle(
        predicted_reference, predicted_direction
    )
    target_relative = _signed_angle(target_reference, target_direction)
    delta = predicted_relative - target_relative
    return torch.rad2deg(torch.atan2(torch.sin(delta), torch.cos(delta)).abs())


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
            joints[..., LEFT_TOE_INDEX, [0, 2]]
            - joints[..., LEFT_ANKLE_INDEX, [0, 2]],
            joints[..., RIGHT_TOE_INDEX, [0, 2]]
            - joints[..., RIGHT_ANKLE_INDEX, [0, 2]],
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
    target_body: torch.Tensor,
    frame_mask: torch.Tensor,
    frame_valid_mask: torch.Tensor | None = None,
    fps: float = 20.0,
    minimum_trajectory_speed_mps: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Measure generated and GT root/body/feet heading relationships.

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
        target_body,
        frame_mask,
        frame_valid_mask,
    )
    root_direction = _root_forward_direction(predicted_root)
    gt_root_direction = _root_forward_direction(target_root)
    body_direction, body_valid = _body_forward_direction(
        predicted_root, predicted_body
    )
    feet_direction, feet_valid = _foot_forward_directions(
        predicted_root, predicted_body
    )
    gt_body_direction, gt_body_valid = _body_forward_direction(
        target_root, target_body
    )
    gt_feet_direction, gt_feet_valid = _foot_forward_directions(
        target_root, target_body
    )
    trajectory_direction, trajectory_valid = _trajectory_forward_direction(
        target_root,
        frame_valid_mask,
        fps=float(fps),
        minimum_speed_mps=float(minimum_trajectory_speed_mps),
    )
    return {
        "root_gt_root_heading_angle_deg": _mean_angle_degrees(
            root_direction,
            gt_root_direction,
            frame_mask,
        ),
        "body_gt_body_heading_angle_deg": _mean_angle_degrees(
            body_direction,
            gt_body_direction,
            frame_mask & body_valid & gt_body_valid,
        ),
        "feet_gt_feet_heading_angle_deg": _mean_angle_degrees(
            feet_direction,
            gt_feet_direction,
            frame_mask[..., None] & feet_valid & gt_feet_valid,
        ),
        "root_trajectory_heading_angle_deg": _mean_angle_degrees(
            root_direction,
            trajectory_direction,
            frame_mask & trajectory_valid,
        ),
        "root_body_heading_angle_deg": _mean_angle_degrees(
            root_direction,
            body_direction,
            frame_mask & body_valid,
        ),
        "root_feet_heading_angle_deg": _mean_angle_degrees(
            root_direction[..., None, :].expand_as(feet_direction),
            feet_direction,
            frame_mask[..., None] & feet_valid,
        ),
        "gt_root_body_heading_angle_deg": _mean_angle_degrees(
            gt_root_direction,
            gt_body_direction,
            frame_mask & gt_body_valid,
        ),
        "gt_root_feet_heading_angle_deg": _mean_angle_degrees(
            gt_root_direction[..., None, :].expand_as(gt_feet_direction),
            gt_feet_direction,
            frame_mask[..., None] & gt_feet_valid,
        ),
        "root_feet_reverse_ratio": _reverse_ratio(
            feet_direction,
            root_direction[..., None, :].expand_as(feet_direction),
            frame_mask[..., None] & feet_valid,
        ),
    }


def compute_rollout_heading_metrics(
    *,
    predicted_root: torch.Tensor,
    target_root: torch.Tensor,
    predicted_body: torch.Tensor,
    target_body: torch.Tensor,
    frame_mask: torch.Tensor,
    cold_frames: int = 4,
) -> dict[str, torch.Tensor]:
    """Measure the compact cold-start and full-rollout heading contract.

    ``body_rel`` and ``feet_rel`` compare the generated-vs-GT relative
    direction to Root. They deliberately do not force the rendered body or
    feet to point in the same direction as Root, which would incorrectly
    penalize side steps, backward walking, and turns.
    """

    if int(cold_frames) <= 0:
        raise ValueError("cold_frames must be positive")
    _validate_inputs(
        predicted_root,
        target_root,
        predicted_body,
        target_body,
        frame_mask,
        frame_mask,
    )
    root = _root_forward_direction(predicted_root)
    target_root_direction = _root_forward_direction(target_root)
    body, body_valid = _body_forward_direction(predicted_root, predicted_body)
    target_body_direction, target_body_valid = _body_forward_direction(
        target_root, target_body
    )
    feet, feet_valid = _foot_forward_directions(predicted_root, predicted_body)
    target_feet, target_feet_valid = _foot_forward_directions(
        target_root, target_body
    )

    root_angle = _unsigned_angle_degrees(root, target_root_direction)
    body_angle = _unsigned_angle_degrees(body, target_body_direction)
    feet_angle = _unsigned_angle_degrees(feet, target_feet)
    body_relative = _relative_heading_error_degrees(
        root,
        body,
        target_root_direction,
        target_body_direction,
    )
    feet_relative = _relative_heading_error_degrees(
        root[..., None, :].expand_as(feet),
        feet,
        target_root_direction[..., None, :].expand_as(target_feet),
        target_feet,
    )

    root_mask = frame_mask
    body_mask = frame_mask & body_valid & target_body_valid
    feet_mask = frame_mask[..., None] & feet_valid & target_feet_valid
    cold_mask = torch.zeros_like(frame_mask)
    cold_mask[:, : min(int(cold_frames), frame_mask.shape[1])] = True
    cold_root_mask = root_mask & cold_mask
    cold_body_mask = body_mask & cold_mask
    cold_feet_mask = feet_mask & cold_mask[..., None]

    normalized_root = torch.nn.functional.normalize(root.float(), dim=-1)
    normalized_target_root = torch.nn.functional.normalize(
        target_root_direction.float(), dim=-1
    )
    root_antipodal = (
        (normalized_root * normalized_target_root).sum(dim=-1) < -0.9
    ).float()

    normalized_feet = torch.nn.functional.normalize(feet.float(), dim=-1)
    normalized_target_feet = torch.nn.functional.normalize(
        target_feet.float(), dim=-1
    )
    predicted_reverse = (
        normalized_feet
        * normalized_root[..., None, :].expand_as(normalized_feet)
    ).sum(dim=-1) < 0.0
    target_reverse = (
        normalized_target_feet
        * normalized_target_root[..., None, :].expand_as(
            normalized_target_feet
        )
    ).sum(dim=-1) < 0.0
    excess_reverse = (predicted_reverse & ~target_reverse).float()

    return {
        "cold_root_deg": _masked_scalar(
            root_angle, cold_root_mask, reduction="mean"
        ),
        "cold_root_max": _masked_scalar(
            root_angle, cold_root_mask, reduction="max"
        ),
        "cold_root_anti": _masked_scalar(
            root_antipodal, cold_root_mask, reduction="mean"
        ),
        "cold_body_deg": _masked_scalar(
            body_angle, cold_body_mask, reduction="mean"
        ),
        "cold_feet_deg": _masked_scalar(
            feet_angle, cold_feet_mask, reduction="mean"
        ),
        "roll_root_deg": _masked_scalar(
            root_angle, root_mask, reduction="mean"
        ),
        "roll_root_p95": _masked_scalar(
            root_angle, root_mask, reduction="p95"
        ),
        "roll_root_max": _masked_scalar(
            root_angle, root_mask, reduction="max"
        ),
        "roll_root_anti": _masked_scalar(
            root_antipodal, root_mask, reduction="mean"
        ),
        "roll_body_deg": _masked_scalar(
            body_angle, body_mask, reduction="mean"
        ),
        "roll_feet_deg": _masked_scalar(
            feet_angle, feet_mask, reduction="mean"
        ),
        "roll_body_rel": _masked_scalar(
            body_relative, body_mask, reduction="mean"
        ),
        "roll_body_rel_max": _masked_scalar(
            body_relative, body_mask, reduction="max"
        ),
        "roll_feet_rel": _masked_scalar(
            feet_relative, feet_mask, reduction="mean"
        ),
        "roll_feet_rel_max": _masked_scalar(
            feet_relative, feet_mask, reduction="max"
        ),
        "roll_feet_rev": _masked_scalar(
            excess_reverse, feet_mask, reduction="mean"
        ),
    }


__all__ = ["compute_heading_metrics", "compute_rollout_heading_metrics"]
