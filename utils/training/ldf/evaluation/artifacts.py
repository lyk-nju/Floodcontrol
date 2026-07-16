"""FloodNet-style artifact layout for LDF validation generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from utils.visualization.motion_video import render_motion_video


ARTIFACT_NAMES = (
    "text",
    "token",
    "feature",
    "traj_xz",
    "traj_mask",
    "frames",
    "metrics",
    "video",
    "composite",
)


def evaluation_artifact_dirs(
    save_dir: str | Path,
    dataset: str,
    probe: str,
    step_tag: str,
) -> dict[str, Path]:
    base = Path(save_dir) / str(dataset)
    return {
        name: base / name / str(probe) / str(step_tag)
        for name in ARTIFACT_NAMES
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if torch.is_tensor(value):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_value(payload), indent=2, sort_keys=True))
    temporary.replace(destination)
    return destination


def _trajectory_panel(
    target_xz: np.ndarray,
    predicted_xz: np.ndarray,
    *,
    frame_index: int,
    size: tuple[int, int] = (480, 480),
) -> Image.Image:
    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    points = np.concatenate([target_xz, predicted_xz], axis=0)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.maximum(upper - lower, 1e-4)
    padding = 40
    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])

    def project(path: np.ndarray) -> list[tuple[float, float]]:
        center = 0.5 * (lower + upper)
        x = width / 2.0 + (path[:, 0] - center[0]) * scale
        y = height / 2.0 - (path[:, 1] - center[1]) * scale
        return list(zip(x.tolist(), y.tolist()))

    target_points = project(target_xz)
    predicted_points = project(predicted_xz)
    if len(target_points) > 1:
        draw.line(target_points, fill=(20, 150, 20), width=4)
    if len(predicted_points) > 1:
        draw.line(predicted_points, fill=(210, 40, 40), width=4)
    index = min(max(int(frame_index), 0), len(target_points) - 1)
    for point, color in (
        (target_points[index], (20, 150, 20)),
        (predicted_points[index], (210, 40, 40)),
    ):
        x, y = point
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    draw.text((16, 14), "Dense XZ: target (green) / generated (red)", fill="black")
    return image


def render_comparison_video(
    *,
    target_root: torch.Tensor,
    target_body: torch.Tensor,
    predicted_root: torch.Tensor,
    predicted_body: torch.Tensor,
    predicted_video_path: str | Path,
    composite_path: str | Path,
    caption: str,
    fps: float,
) -> None:
    """Render generated motion plus a GT/generated/trajectory comparison video."""

    predicted_video_path = Path(predicted_video_path)
    composite_path = Path(composite_path)
    predicted_video_path.parent.mkdir(parents=True, exist_ok=True)
    composite_path.parent.mkdir(parents=True, exist_ok=True)
    render_motion_video(
        predicted_root,
        predicted_body,
        predicted_video_path,
        fps=fps,
    )

    with tempfile.TemporaryDirectory(prefix="floodcontrol_eval_") as temporary:
        target_path = Path(temporary) / "target.mp4"
        render_motion_video(target_root, target_body, target_path, fps=fps)
        target_reader = imageio.get_reader(str(target_path))
        predicted_reader = imageio.get_reader(str(predicted_video_path))
        writer = imageio.get_writer(str(composite_path), fps=float(fps))
        target_xz = target_root.detach().cpu().numpy()[:, [0, 2]]
        predicted_xz = predicted_root.detach().cpu().numpy()[:, [0, 2]]
        try:
            for frame_index, (target_frame, predicted_frame) in enumerate(
                zip(target_reader, predicted_reader)
            ):
                target_image = Image.fromarray(target_frame).convert("RGB")
                predicted_image = Image.fromarray(predicted_frame).convert("RGB")
                panel = _trajectory_panel(
                    target_xz,
                    predicted_xz,
                    frame_index=frame_index,
                    size=target_image.size,
                )
                canvas = Image.new(
                    "RGB",
                    (
                        target_image.width + predicted_image.width + panel.width,
                        target_image.height + 32,
                    ),
                    "white",
                )
                canvas.paste(target_image, (0, 32))
                canvas.paste(predicted_image, (target_image.width, 32))
                canvas.paste(panel, (target_image.width + predicted_image.width, 32))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 8), "Ground truth", fill="black")
                draw.text((target_image.width + 12, 8), "Generated", fill="black")
                draw.text(
                    (target_image.width * 2 + 12, 8),
                    str(caption)[:80],
                    fill="black",
                )
                writer.append_data(np.asarray(canvas, dtype=np.uint8))
        finally:
            target_reader.close()
            predicted_reader.close()
            writer.close()


def save_dense_xz_sample(
    *,
    save_dir: str | Path,
    dataset: str,
    probe: str,
    step_tag: str,
    sample_id: str,
    caption: str,
    normalized_root: torch.Tensor,
    normalized_latent: torch.Tensor,
    predicted_root: torch.Tensor,
    predicted_body: torch.Tensor,
    target_root: torch.Tensor,
    target_body: torch.Tensor,
    trajectory_mask: torch.Tensor,
    prompt_change_frames: np.ndarray,
    record: dict[str, Any],
    render: bool,
    fps: float,
) -> dict[str, Path]:
    dirs = evaluation_artifact_dirs(save_dir, dataset, probe, step_tag)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    (dirs["text"] / f"{sample_id}.txt").write_text(str(caption))
    np.savez(
        dirs["token"] / f"{sample_id}.npz",
        root_motion=normalized_root.detach().cpu().float().numpy(),
        latent_motion=normalized_latent.detach().cpu().float().numpy(),
    )
    np.savez(
        dirs["feature"] / f"{sample_id}.npz",
        root_motion=predicted_root.detach().cpu().float().numpy(),
        body_motion=predicted_body.detach().cpu().float().numpy(),
    )
    np.save(
        dirs["traj_xz"] / f"{sample_id}.npy",
        target_root.detach().cpu().float().numpy()[:, [0, 2]],
    )
    np.save(
        dirs["traj_mask"] / f"{sample_id}.npy",
        trajectory_mask.detach().cpu().bool().numpy(),
    )
    np.save(dirs["frames"] / f"{sample_id}.npy", prompt_change_frames)
    write_json(dirs["metrics"] / f"{sample_id}.json", record)

    if render:
        render_comparison_video(
            target_root=target_root,
            target_body=target_body,
            predicted_root=predicted_root,
            predicted_body=predicted_body,
            predicted_video_path=dirs["video"] / f"{sample_id}.mp4",
            composite_path=dirs["composite"] / f"{sample_id}.mp4",
            caption=caption,
            fps=fps,
        )
    return dirs


__all__ = [
    "evaluation_artifact_dirs",
    "render_comparison_video",
    "save_dense_xz_sample",
    "write_json",
]
