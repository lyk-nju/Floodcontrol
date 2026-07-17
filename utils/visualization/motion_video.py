"""Lightweight HumanML22 motion video rendering.

``render_joint_video`` consumes world-space joints and owns only projection
and rasterization. ``render_motion_video`` is the physical root5/body265
adapter. Directory traversal, evaluation artifact layout, and video comparison
belong to their task-specific callers rather than this module.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from utils.motion_process import BODY_DIM, NUM_JOINTS, ROOT_DIM, recover_joint_positions
from utils.visualization.skeleton import HUMANML22_CHAINS, HUMANML22_CHAIN_COLORS


DEFAULT_IMAGE_SIZE = (480, 480)
DEFAULT_BACKGROUND_COLOR = (255, 255, 255)
DEFAULT_JOINT_COLOR = (0, 100, 255)
TARGET_TRAJECTORY_COLOR = (255, 0, 0)
GENERATED_TRAJECTORY_COLOR = (0, 0, 255)


def _numpy_float32(value: np.ndarray | torch.Tensor, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_image_size(image_size: tuple[int, int]) -> tuple[int, int]:
    if len(image_size) != 2:
        raise ValueError("image_size must be a (width, height) pair")
    width, height = image_size
    if isinstance(width, bool) or isinstance(height, bool):
        raise TypeError("image dimensions must be integers")
    if int(width) != width or int(height) != height:
        raise TypeError("image dimensions must be integers")
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if width % 2 or height % 2:
        raise ValueError("video image dimensions must be even for H.264 encoding")
    return width, height


def _project_scene(
    joint_positions: np.ndarray,
    *,
    trajectory_xz: np.ndarray | None,
    width: int,
    height: int,
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Project motion and ground-plane paths with one fixed orthographic camera."""
    elevation = -math.pi / 10.0
    azimuth = -3.0 * math.pi / 4.0
    view = np.asarray(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
            math.cos(elevation) * math.sin(azimuth),
        ),
        dtype=np.float32,
    )
    view /= np.linalg.norm(view)
    world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    camera_right = np.cross(view, world_up)
    camera_right /= np.linalg.norm(camera_right)
    camera_up = np.cross(camera_right, view)
    camera_up /= np.linalg.norm(camera_up)

    root_ground = np.stack(
        (
            joint_positions[:, 0, 0],
            np.zeros(joint_positions.shape[0], dtype=np.float32),
            joint_positions[:, 0, 2],
        ),
        axis=-1,
    )
    scene_points = [joint_positions.reshape(-1, 3), root_ground]
    trajectory_ground = None
    if trajectory_xz is not None:
        trajectory_ground = np.stack(
            (
                trajectory_xz[:, 0],
                np.zeros(trajectory_xz.shape[0], dtype=np.float32),
                trajectory_xz[:, 1],
            ),
            axis=-1,
        )
        scene_points.append(trajectory_ground)
    bounds_points = np.concatenate(scene_points, axis=0)
    horizontal = bounds_points @ camera_right
    vertical = bounds_points @ camera_up
    horizontal_min, horizontal_max = float(horizontal.min()), float(horizontal.max())
    vertical_min, vertical_max = float(vertical.min()), float(vertical.max())
    horizontal_range = max(horizontal_max - horizontal_min, 1e-6)
    vertical_range = max(vertical_max - vertical_min, 1e-6)
    usable_width = width - 2 * padding
    usable_height = height - 2 * padding
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("padding leaves no drawable image area")
    scale = min(usable_width / horizontal_range, usable_height / vertical_range)

    horizontal_center = 0.5 * (horizontal_min + horizontal_max)
    vertical_center = 0.5 * (vertical_min + vertical_max)
    def project(points: np.ndarray) -> np.ndarray:
        projected_horizontal = points @ camera_right
        projected_vertical = points @ camera_up
        screen_x = width / 2.0 + (projected_horizontal - horizontal_center) * scale
        screen_y = height / 2.0 - (projected_vertical - vertical_center) * scale
        return np.stack((screen_x, screen_y), axis=-1)

    return (
        project(joint_positions),
        project(root_ground),
        None if trajectory_ground is None else project(trajectory_ground),
    )


def render_joint_video(
    joint_positions: np.ndarray | torch.Tensor,
    output_path: str | Path,
    *,
    fps: float = 20.0,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    padding: int = 24,
    bone_width: int = 4,
    joint_radius: int = 3,
    traj_xz: np.ndarray | torch.Tensor | None = None,
    traj_mask: np.ndarray | torch.Tensor | None = None,
    show_full_trajectory: bool = False,
    show_generated_trajectory: bool = False,
) -> None:
    """Render world-space HumanML22 joints as a fixed-camera MP4.

    Args:
        joint_positions: Physical world joints ``[F,22,3]``.
        output_path: Destination video path.
        fps: Positive output frame rate.
        image_size: Even ``(width, height)`` required by H.264 encoders.
        padding: Minimum projection margin in pixels.
        bone_width: Bone line width in pixels.
        joint_radius: Joint circle radius in pixels.
        traj_xz: Optional physical target/conditioning path ``[F,2]`` as x/z.
        traj_mask: Optional observed-frame mask ``[F]`` for the target path.
        show_full_trajectory: Draw the complete target path from frame zero;
            otherwise reveal it up to the current frame.
        show_generated_trajectory: Draw the generated root path up to the
            current frame using the same timestamps as ``traj_mask``.
    """
    joints = _numpy_float32(joint_positions, name="joint_positions")
    if joints.ndim != 3 or tuple(joints.shape[1:]) != (NUM_JOINTS, 3):
        raise ValueError(
            f"joint_positions must be [F,{NUM_JOINTS},3], got {tuple(joints.shape)}"
        )
    if joints.shape[0] == 0:
        raise ValueError("joint_positions must contain at least one frame")
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    width, height = _validate_image_size(image_size)
    for name, value in (
        ("padding", padding),
        ("bone_width", bone_width),
        ("joint_radius", joint_radius),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    padding, bone_width, joint_radius = int(padding), int(bone_width), int(joint_radius)

    trajectory = None
    trajectory_mask = None
    if traj_xz is not None:
        trajectory = _numpy_float32(traj_xz, name="traj_xz")
        if trajectory.ndim != 2 or trajectory.shape != (joints.shape[0], 2):
            raise ValueError(
                f"traj_xz must be [F,2] and match motion length, got {tuple(trajectory.shape)}"
            )
        if traj_mask is None:
            trajectory_mask = np.ones(joints.shape[0], dtype=bool)
        else:
            if isinstance(traj_mask, torch.Tensor):
                traj_mask = traj_mask.detach().cpu().numpy()
            trajectory_mask = np.asarray(traj_mask)
            if trajectory_mask.shape != (joints.shape[0],):
                raise ValueError(
                    "traj_mask must be [F] and match motion length, "
                    f"got {tuple(trajectory_mask.shape)}"
                )
            trajectory_mask = trajectory_mask.astype(bool, copy=False)
    elif traj_mask is not None:
        raise ValueError("traj_mask requires traj_xz")
    if show_generated_trajectory and trajectory is None:
        raise ValueError("show_generated_trajectory requires traj_xz")

    projected, projected_root, projected_trajectory = _project_scene(
        joints,
        trajectory_xz=trajectory,
        width=width,
        height=height,
        padding=padding,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(destination), fps=fps)
    try:
        for frame_index, frame_positions in enumerate(projected):
            image = Image.new("RGB", (width, height), DEFAULT_BACKGROUND_COLOR)
            draw = ImageDraw.Draw(image)
            if projected_trajectory is not None and trajectory_mask is not None:
                target_limit = len(trajectory_mask) if show_full_trajectory else frame_index + 1
                target_indices = np.flatnonzero(trajectory_mask[:target_limit])
                target_points = [
                    tuple(projected_trajectory[index]) for index in target_indices
                ]
                if len(target_points) > 1:
                    draw.line(target_points, fill=TARGET_TRAJECTORY_COLOR, width=3)
                for point in target_points:
                    x, y = point
                    draw.ellipse(
                        (x - 2, y - 2, x + 2, y + 2),
                        fill=TARGET_TRAJECTORY_COLOR,
                    )

                if show_generated_trajectory:
                    generated_indices = np.flatnonzero(
                        trajectory_mask[: frame_index + 1]
                    )
                    generated_points = [
                        tuple(projected_root[index]) for index in generated_indices
                    ]
                    if len(generated_points) > 1:
                        draw.line(
                            generated_points,
                            fill=GENERATED_TRAJECTORY_COLOR,
                            width=3,
                        )
                    if generated_points:
                        x, y = generated_points[-1]
                        draw.ellipse(
                            (x - 5, y - 5, x + 5, y + 5),
                            fill=GENERATED_TRAJECTORY_COLOR,
                        )

                draw.line((12, 17, 34, 17), fill=TARGET_TRAJECTORY_COLOR, width=3)
                draw.text((40, 10), "target", fill=(0, 0, 0))
                if show_generated_trajectory:
                    draw.line(
                        (104, 17, 126, 17),
                        fill=GENERATED_TRAJECTORY_COLOR,
                        width=3,
                    )
                    draw.text((132, 10), "generated", fill=(0, 0, 0))
            for chain_index, chain in enumerate(HUMANML22_CHAINS):
                color = HUMANML22_CHAIN_COLORS[chain_index]
                for start, end in zip(chain[:-1], chain[1:], strict=True):
                    draw.line(
                        (tuple(frame_positions[start]), tuple(frame_positions[end])),
                        fill=color,
                        width=bone_width,
                    )
            for x, y in frame_positions:
                draw.ellipse(
                    (
                        x - joint_radius,
                        y - joint_radius,
                        x + joint_radius,
                        y + joint_radius,
                    ),
                    fill=DEFAULT_JOINT_COLOR,
                )
            writer.append_data(np.asarray(image, dtype=np.uint8))
    finally:
        writer.close()


def render_motion_video(
    root_motion: np.ndarray | torch.Tensor,
    body_motion: np.ndarray | torch.Tensor,
    output_path: str | Path,
    *,
    fps: float = 20.0,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    traj_xz: np.ndarray | torch.Tensor | None = None,
    traj_mask: np.ndarray | torch.Tensor | None = None,
    show_full_trajectory: bool = False,
    show_generated_trajectory: bool = False,
) -> None:
    """Recover and render one physical root5/body265 motion and optional route."""
    root = torch.from_numpy(_numpy_float32(root_motion, name="root_motion"))
    body = torch.from_numpy(_numpy_float32(body_motion, name="body_motion"))
    if root.ndim != 2 or root.shape[-1] != ROOT_DIM:
        raise ValueError(f"root_motion must be [F,{ROOT_DIM}]")
    if body.ndim != 2 or body.shape[-1] != BODY_DIM:
        raise ValueError(f"body_motion must be [F,{BODY_DIM}]")
    if root.shape[0] != body.shape[0]:
        raise ValueError("root_motion and body_motion must share frame length")
    joints = recover_joint_positions(root, body)
    render_joint_video(
        joints,
        output_path,
        fps=fps,
        image_size=image_size,
        traj_xz=traj_xz,
        traj_mask=traj_mask,
        show_full_trajectory=show_full_trajectory,
        show_generated_trajectory=show_generated_trajectory,
    )


__all__ = ["render_joint_video", "render_motion_video"]
