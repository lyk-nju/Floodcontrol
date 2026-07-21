"""Dense XZ trajectory and motion-quality metrics for LDF evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from utils.motion_process import BODY_DIM, ROOT_DIM, recover_joint_positions


def _motion_xz(value: torch.Tensor | np.ndarray, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu()
    if tensor.ndim != 2 or tensor.shape[-1] not in (2, ROOT_DIM):
        raise ValueError(f"{name} must be [F,2] or [F,{ROOT_DIM}]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor if tensor.shape[-1] == 2 else tensor[:, [0, 2]]


def compute_dense_xz_metrics(
    predicted_root: torch.Tensor | np.ndarray,
    target_root: torch.Tensor | np.ndarray,
    *,
    valid_mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return the minimal time-aligned dense-XZ control metrics in metres."""

    predicted = _motion_xz(predicted_root, name="predicted_root")
    target = _motion_xz(target_root, name="target_root")
    frames = min(len(predicted), len(target))
    if frames <= 0:
        raise ValueError("dense XZ evaluation requires at least one aligned frame")
    predicted, target = predicted[:frames], target[:frames]
    if valid_mask is None:
        mask = torch.ones(frames, dtype=torch.bool)
    else:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool).detach().cpu().reshape(-1)
        if len(mask) < frames:
            raise ValueError("valid_mask is shorter than the aligned trajectory")
        mask = mask[:frames]
    if not bool(mask.any()):
        raise ValueError("dense XZ evaluation mask contains no valid frames")

    delta = predicted - target
    distance = delta.norm(dim=-1)
    selected_distance = distance[mask]
    last_index = int(mask.nonzero(as_tuple=False)[-1].item())
    return {
        "frames": int(frames),
        "ade": float(selected_distance.mean().item()),
        "fde": float(distance[last_index].item()),
        "max_error": float(selected_distance.max().item()),
    }


def compute_foot_skating_ratio(
    root_motion: torch.Tensor | np.ndarray,
    body_motion: torch.Tensor | np.ndarray,
    *,
    fps: float = 20.0,
    height_threshold: float = 0.05,
    speed_threshold: float = 0.50,
) -> float:
    """Measure HumanML22 foot sliding while either foot is near the floor."""

    root = torch.as_tensor(root_motion, dtype=torch.float32).detach().cpu()
    body = torch.as_tensor(body_motion, dtype=torch.float32).detach().cpu()
    if root.ndim != 2 or root.shape[-1] != ROOT_DIM:
        raise ValueError(f"root_motion must be [F,{ROOT_DIM}]")
    if body.ndim != 2 or body.shape[-1] != BODY_DIM:
        raise ValueError(f"body_motion must be [F,{BODY_DIM}]")
    if root.shape[0] != body.shape[0]:
        raise ValueError("root_motion and body_motion must share frame length")
    if root.shape[0] < 2:
        return float("nan")
    joints = recover_joint_positions(root, body).numpy()
    feet = joints[:, [10, 11]]
    planar_speed = (
        np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
        * float(fps)
    )
    contact = np.logical_and(
        feet[:-1, :, 1] < float(height_threshold),
        feet[1:, :, 1] < float(height_threshold),
    )
    skating = np.logical_and(contact, planar_speed > float(speed_threshold)).any(axis=1)
    return float(skating.mean())


def summarize_dense_xz_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate finite scalar and per-segment metrics across generated samples."""

    summary: dict[str, Any] = {"num_samples": int(len(records))}
    scalar_keys = (
        "ade",
        "fde",
        "max_error",
        "foot_skating_ratio",
        "root_boundary_jump_mean",
        "joint_boundary_jump_mean",
        "root_gt_root_heading_angle_deg",
        "body_gt_body_heading_angle_deg",
        "feet_gt_feet_heading_angle_deg",
        "root_trajectory_heading_angle_deg",
        "root_body_heading_angle_deg",
        "root_feet_heading_angle_deg",
        "gt_root_body_heading_angle_deg",
        "gt_root_feet_heading_angle_deg",
        "root_feet_reverse_ratio",
    )
    for key in scalar_keys:
        values = np.asarray(
            [float(record[key]) for record in records if key in record],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size:
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std())

    return summary


__all__ = [
    "compute_dense_xz_metrics",
    "compute_foot_skating_ratio",
    "summarize_dense_xz_records",
]
