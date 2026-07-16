"""Artifact serialization and visualization for BodyVAE evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from utils.motion_process import recover_joint_positions
from utils.visualization import render_joint_video

from .reconstruction import MotionSample, ReconstructionResult


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError("only scalar tensors may be serialized to JSON")
        return value.item()
    return value


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def validate_model_id(model_id: str) -> str:
    """Validate the explicit directory identity used for one VAE checkpoint."""

    model_id = str(model_id).strip()
    if not model_id or model_id in {".", ".."}:
        raise ValueError("model.model_id must be a non-empty directory name")
    if Path(model_id).name != model_id:
        raise ValueError(
            "model.model_id must be one directory name without path separators"
        )
    return model_id


def output_paths(
    output_root: Path,
    dataset_name: str,
    model_id: str,
    sample_id: str,
) -> dict[str, Path]:
    root = output_root / dataset_name / validate_model_id(model_id)
    return {
        "original_video": root / "video" / "original" / f"{sample_id}.mp4",
        "reconstruction_video": root
        / "video"
        / "reconstruction"
        / f"{sample_id}.mp4",
        "original_motion": root / "motion" / "original" / f"{sample_id}.npz",
        "reconstruction_motion": root
        / "motion"
        / "reconstruction"
        / f"{sample_id}.npz",
        "metrics": root / "metrics" / f"{sample_id}.json",
    }


def save_sample_outputs(
    sample: MotionSample,
    result: ReconstructionResult,
    metrics: Mapping[str, object],
    *,
    output_root: Path,
    dataset_name: str,
    model_id: str,
    render_video: bool,
    render_fps: int,
) -> dict[str, str]:
    """Persist one source/reconstruction pair and optional videos."""

    paths = output_paths(output_root, dataset_name, model_id, sample.sample_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    original_joints = recover_joint_positions(sample.root_motion, sample.body_motion)
    reconstructed_continuous = result.streamed_body.continuous_body[0]
    reconstructed_joints = recover_joint_positions(
        sample.root_motion, reconstructed_continuous
    )
    np.savez_compressed(
        paths["original_motion"],
        root_motion=sample.root_motion.numpy(),
        body_motion=sample.body_motion.numpy(),
        global_joints=original_joints.numpy(),
        fps=np.float32(sample.fps),
        sample_id=sample.sample_id,
        dataset=sample.dataset,
    )
    contact_logits = result.streamed_body.contact_logits[0]
    reconstructed_body = torch.cat(
        [reconstructed_continuous, (contact_logits.sigmoid() >= 0.5).float()],
        dim=-1,
    )
    reconstruction_payload = {
        "root_motion": sample.root_motion.numpy(),
        "body_motion": reconstructed_body.numpy(),
        "continuous_body": reconstructed_continuous.numpy(),
        "contact_logits": contact_logits.numpy(),
        "contact_probability": contact_logits.sigmoid().numpy(),
        "global_joints": reconstructed_joints.numpy(),
        "posterior_mu": result.posterior_mu[0].numpy(),
        "local_root_motion": result.local_root_motion[0].numpy(),
        "local_root_valid_mask": result.local_root_valid_mask[0].numpy(),
        "fps": np.float32(sample.fps),
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "protocol": result.protocol,
    }
    if result.rolling_trace is not None:
        reconstruction_payload.update(
            {
                f"rolling_{name}": value.numpy()
                for name, value in result.rolling_trace.items()
            }
        )
    if result.reference_stream_body is not None:
        reconstruction_payload.update(
            {
                "reference_stream_continuous_body": (
                    result.reference_stream_body.continuous_body[0].numpy()
                ),
                "reference_stream_contact_logits": (
                    result.reference_stream_body.contact_logits[0].numpy()
                ),
            }
        )
    np.savez_compressed(paths["reconstruction_motion"], **reconstruction_payload)
    write_json(paths["metrics"], metrics)
    if render_video:
        render_joint_video(original_joints, paths["original_video"], fps=render_fps)
        render_joint_video(
            reconstructed_joints,
            paths["reconstruction_video"],
            fps=render_fps,
        )
    return {name: str(path) for name, path in paths.items()}


__all__ = [
    "output_paths",
    "save_sample_outputs",
    "validate_model_id",
    "write_json",
]
