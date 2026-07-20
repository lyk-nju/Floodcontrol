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


def _resample_path(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    lengths = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(lengths, dtype=np.float32)]
    )
    if float(cumulative[-1]) < 1e-8:
        return np.repeat(points[:1], count, axis=0)
    samples = np.linspace(0.0, float(cumulative[-1]), count, dtype=np.float32)
    return np.stack(
        [
            np.interp(samples, cumulative, points[:, 0]),
            np.interp(samples, cumulative, points[:, 1]),
        ],
        axis=-1,
    ).astype(np.float32)


def _path_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    count = min(max(len(predicted), len(target)), 256)
    pred = _resample_path(predicted.numpy(), count)
    ref = _resample_path(target.numpy(), count)
    arc_ade = float(np.linalg.norm(pred - ref, axis=-1).mean())
    distances = np.linalg.norm(pred[:, None] - ref[None], axis=-1)
    chamfer = float(
        0.5 * (distances.min(axis=1).mean() + distances.min(axis=0).mean())
    )
    return {"path_arc_ade": arc_ade, "path_chamfer": chamfer}


def compute_dense_xz_metrics(
    predicted_root: torch.Tensor | np.ndarray,
    target_root: torch.Tensor | np.ndarray,
    *,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    segment_frames: int = 20,
) -> dict[str, Any]:
    """Compare time-aligned dense root XZ trajectories in physical metres."""

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
    squared_distance = delta.square().sum(dim=-1)
    selected_distance = distance[mask]
    last_index = int(mask.nonzero(as_tuple=False)[-1].item())
    result: dict[str, Any] = {
        "frames": int(frames),
        "masked_ratio": float(mask.float().mean().item()),
        "ade": float(selected_distance.mean().item()),
        "fde": float(distance[last_index].item()),
        "mse": float(squared_distance[mask].mean().item()),
        "traj_fail_20cm": float((selected_distance > 0.20).any().item()),
        "traj_fail_50cm": float((selected_distance > 0.50).any().item()),
        "frame_fail_20cm": float((selected_distance > 0.20).float().mean().item()),
        "frame_fail_50cm": float((selected_distance > 0.50).float().mean().item()),
    }
    result.update(_path_metrics(predicted[mask], target[mask]))

    segment = int(segment_frames)
    if segment <= 0:
        raise ValueError("segment_frames must be positive")
    result["segment_mse"] = [
        (
            float(squared_distance[start:end][mask[start:end]].mean().item())
            if bool(mask[start:end].any())
            else None
        )
        for start in range(0, frames, segment)
        for end in [min(start + segment, frames)]
    ]
    result["prefix_mse"] = [
        float(squared_distance[:end][mask[:end]].mean().item())
        for end in range(segment, frames + 1, segment)
        if bool(mask[:end].any())
    ]
    if frames >= 3:
        acceleration = predicted[2:] - 2.0 * predicted[1:-1] + predicted[:-2]
        result["traj_jitter"] = float(
            acceleration.square().sum(dim=-1).mean().item()
        )
    else:
        result["traj_jitter"] = float("nan")
    return result


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
        "mse",
        "traj_fail_20cm",
        "traj_fail_50cm",
        "frame_fail_20cm",
        "frame_fail_50cm",
        "traj_jitter",
        "path_arc_ade",
        "path_chamfer",
        "foot_skating_ratio",
        "root_boundary_jump_mean",
        "joint_boundary_jump_mean",
        "root_gt_heading_angle_deg",
        "root_gt_trajectory_heading_angle_deg",
        "root_body_heading_angle_deg",
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

    for key in ("segment_mse", "prefix_mse"):
        width = max((len(record.get(key, [])) for record in records), default=0)
        slots = []
        for index in range(width):
            values = [
                float(record[key][index])
                for record in records
                if index < len(record.get(key, []))
                and record[key][index] is not None
                and np.isfinite(float(record[key][index]))
            ]
            slots.append(float(np.mean(values)) if values else None)
        if slots:
            summary[f"{key}_per_slot"] = slots
    return summary


__all__ = [
    "compute_dense_xz_metrics",
    "compute_foot_skating_ratio",
    "summarize_dense_xz_records",
]
