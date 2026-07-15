"""Shared implementation for direct-stream and rolling VAE reconstruction.

The deterministic posterior mean stands in for an LDF-produced body token.
The direct task keeps one decoder state for the complete sequence. The rolling
task starts a fresh state at every commit, replays only the retained history,
and then decodes the current token. The explicit root is not reconstructed by
the VAE and is shared by the original and reconstructed visualization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.babel import BABELDataset
from datasets.humanml3d import HumanML3DDataset
from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import (
    BODY_CONTINUOUS_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    NUM_JOINTS,
    BodyPrediction,
)
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
)
from utils.motion_process import (
    recover_joint_positions,
    recover_local_root,
    rotation_to_matrix,
)
from utils.training.vae.checkpoint import load_vae_checkpoint
from utils.visualization import render_joint_video


STREAM_PROTOCOL = "deterministic-mu-stream-decode-v1"
ROLLING_PROTOCOL = "deterministic-mu-history-replay-v1"
DATASET_TYPES = {
    "humanml3d": HumanML3DDataset,
    "babel": BABELDataset,
}


@dataclass(frozen=True)
class MotionSample:
    sample_id: str
    dataset: str
    root_motion: torch.Tensor
    body_motion: torch.Tensor
    body_feature_valid_mask: torch.Tensor
    previous_root_frame: torch.Tensor | None
    fps: float


@dataclass(frozen=True)
class ReconstructionResult:
    protocol: str
    posterior_mu: torch.Tensor
    local_root_motion: torch.Tensor
    local_root_valid_mask: torch.Tensor
    streamed_body: BodyPrediction
    offline_body: BodyPrediction
    stream_offline_max_abs: float
    reference_stream_body: BodyPrediction | None = None
    rolling_reference_max_abs: float | None = None
    rolling_trace: Mapping[str, torch.Tensor] | None = None


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def load_motion_sample(sample: Mapping[str, object], *, expected_fps: float) -> MotionSample:
    """Adapt one already validated full Dataset sample for reconstruction."""

    return MotionSample(
        sample_id=str(sample["name"]),
        dataset=str(sample["dataset"]),
        root_motion=sample["root_motion"],
        body_motion=sample["body_motion"],
        body_feature_valid_mask=sample["body_feature_valid_mask"],
        previous_root_frame=None,
        fps=float(expected_fps),
    )


@torch.no_grad()
def stream_reconstruct(
    model: BodyVAE,
    sample: MotionSample,
    *,
    device: torch.device | str,
    parity_atol: float = 1e-5,
) -> ReconstructionResult:
    device = torch.device(device)
    body = sample.body_motion[None].to(device)
    root = sample.root_motion[None].to(device)
    frame_valid = torch.ones(body.shape[:2], dtype=torch.bool, device=device)
    previous_root = (
        sample.previous_root_frame[None].to(device)
        if sample.previous_root_frame is not None
        else None
    )
    previous_valid = (
        torch.ones(1, dtype=torch.bool, device=device)
        if previous_root is not None
        else None
    )
    posterior_mu = model.encode(body, frame_valid).mu
    local_root, local_valid = recover_local_root(
        root,
        previous_root,
        fps=sample.fps,
        previous_root_valid_mask=previous_valid,
    )
    offline = model.decode(posterior_mu, local_root, local_valid, frame_valid)
    state = model.init_decoder_state(1, device=device, dtype=posterior_mu.dtype)
    continuous_chunks = []
    contact_chunks = []
    for token_index in range(posterior_mu.shape[1]):
        state, prediction = model.decode_step(
            posterior_mu[:, token_index : token_index + 1],
            local_root[:, token_index : token_index + 1],
            local_valid[:, token_index : token_index + 1],
            state,
        )
        continuous_chunks.append(prediction.continuous_body)
        contact_chunks.append(prediction.contact_logits)
    streamed = BodyPrediction(
        continuous_body=torch.cat(continuous_chunks, dim=1),
        contact_logits=torch.cat(contact_chunks, dim=1),
    )
    max_abs = max(
        float((streamed.continuous_body - offline.continuous_body).abs().max()),
        float((streamed.contact_logits - offline.contact_logits).abs().max()),
    )
    if max_abs > float(parity_atol):
        raise RuntimeError(
            f"offline/stream decoder parity failed for {sample.dataset}/{sample.sample_id}: "
            f"max_abs={max_abs:.8g}, tolerance={parity_atol:.8g}"
        )
    return ReconstructionResult(
        protocol=STREAM_PROTOCOL,
        posterior_mu=posterior_mu.cpu(),
        local_root_motion=local_root.cpu(),
        local_root_valid_mask=local_valid.cpu(),
        streamed_body=BodyPrediction(
            streamed.continuous_body.cpu(), streamed.contact_logits.cpu()
        ),
        offline_body=BodyPrediction(
            offline.continuous_body.cpu(), offline.contact_logits.cpu()
        ),
        stream_offline_max_abs=max_abs,
    )


def create_rolling_window(
    posterior_mu: torch.Tensor,
    *,
    commit_index: int,
    history_tokens: int,
) -> dict[str, torch.Tensor | int]:
    """Create a right-aligned history view followed by the current token."""

    if posterior_mu.ndim != 3 or posterior_mu.shape[0] != 1:
        raise ValueError("rolling reconstruction expects posterior_mu [1,T,D]")
    total_tokens = int(posterior_mu.shape[1])
    if not 0 <= int(commit_index) < total_tokens:
        raise ValueError("commit_index is outside the motion token sequence")
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    history_start = max(0, int(commit_index) - int(history_tokens))
    history_end = int(commit_index)
    history_count = history_end - history_start
    capacity = int(history_tokens) + 1
    values = posterior_mu.new_zeros(1, capacity, posterior_mu.shape[-1])
    timeline_position_ids = torch.full(
        (1, capacity), -1, dtype=torch.long, device=posterior_mu.device
    )
    history_mask = torch.zeros(
        1, capacity, dtype=torch.bool, device=posterior_mu.device
    )
    current_mask = torch.zeros_like(history_mask)
    history_slot_start = int(history_tokens) - history_count
    if history_count:
        history_slice = slice(history_slot_start, int(history_tokens))
        values[:, history_slice] = posterior_mu[:, history_start:history_end]
        timeline_position_ids[:, history_slice] = torch.arange(
            history_start, history_end, device=posterior_mu.device
        )
        history_mask[:, history_slice] = True
    values[:, int(history_tokens) : int(history_tokens) + 1] = posterior_mu[
        :, int(commit_index) : int(commit_index) + 1
    ]
    timeline_position_ids[:, int(history_tokens)] = int(commit_index)
    current_mask[:, int(history_tokens)] = True
    if int(timeline_position_ids[0, history_tokens]) != int(commit_index):
        raise AssertionError("the current slot must be the committed token")
    return {
        "values": values,
        "timeline_position_ids": timeline_position_ids,
        "history_mask": history_mask,
        "current_mask": current_mask,
        "window_origin": history_start,
        "history_start": history_start,
        "history_end": history_end,
        "window_end": int(commit_index) + 1,
        "commit_token": int(commit_index),
    }


@torch.no_grad()
def rolling_reconstruct(
    model: BodyVAE,
    sample: MotionSample,
    *,
    device: torch.device | str,
    history_tokens: int = 10,
    commit_tokens: int = 1,
    parity_atol: float = 1e-5,
) -> ReconstructionResult:
    """Decode each token from a fresh state with finite replayed history."""

    if int(commit_tokens) != 1:
        raise ValueError("rolling VAE evaluation currently requires commit_tokens=1")
    if int(history_tokens) <= 0:
        raise ValueError("rolling VAE evaluation requires history_tokens > 0")
    device = torch.device(device)
    body = sample.body_motion[None].to(device)
    root = sample.root_motion[None].to(device)
    frame_valid = torch.ones(body.shape[:2], dtype=torch.bool, device=device)
    previous_root = (
        sample.previous_root_frame[None].to(device)
        if sample.previous_root_frame is not None
        else None
    )
    previous_valid = (
        torch.ones(1, dtype=torch.bool, device=device)
        if previous_root is not None
        else None
    )
    posterior_mu = model.encode(body, frame_valid).mu
    local_root, local_valid = recover_local_root(
        root,
        previous_root,
        fps=sample.fps,
        previous_root_valid_mask=previous_valid,
    )
    offline = model.decode(posterior_mu, local_root, local_valid, frame_valid)

    # Persistent stream is the full-history reference. It uses exactly the same
    # posterior means and GT local-root patches as finite-history rolling.
    reference_continuous = []
    reference_contacts = []
    reference_state = model.init_decoder_state(
        1, device=device, dtype=posterior_mu.dtype
    )
    for token_index in range(posterior_mu.shape[1]):
        reference_state, reference_prediction = model.decode_step(
            posterior_mu[:, token_index : token_index + 1],
            local_root[:, token_index : token_index + 1],
            local_valid[:, token_index : token_index + 1],
            reference_state,
        )
        reference_continuous.append(reference_prediction.continuous_body)
        reference_contacts.append(reference_prediction.contact_logits)
    reference_stream = BodyPrediction(
        continuous_body=torch.cat(reference_continuous, dim=1),
        contact_logits=torch.cat(reference_contacts, dim=1),
    )
    reference_offline_max_abs = max(
        float((reference_stream.continuous_body - offline.continuous_body).abs().max()),
        float((reference_stream.contact_logits - offline.contact_logits).abs().max()),
    )
    if reference_offline_max_abs > float(parity_atol):
        raise RuntimeError(
            f"offline/persistent-stream decoder parity failed for "
            f"{sample.dataset}/{sample.sample_id}: max_abs={reference_offline_max_abs:.8g}, "
            f"tolerance={parity_atol:.8g}"
        )

    continuous_chunks = []
    contact_chunks = []
    cache_window_max_abs = 0.0
    trace_lists: dict[str, list[torch.Tensor | int]] = {
        "timeline_position_ids": [],
        "history_mask": [],
        "current_mask": [],
        "window_origin": [],
        "history_start": [],
        "history_end": [],
        "window_end": [],
        "commit_token": [],
    }
    for commit_index in range(posterior_mu.shape[1]):
        window = create_rolling_window(
            posterior_mu,
            commit_index=commit_index,
            history_tokens=int(history_tokens),
        )
        history_start = int(window["history_start"])
        window_end = int(window["window_end"])

        # This reset is the defining rolling behavior: cache before
        # history_start is discarded, retained tokens are replayed, and only
        # the current token's four output frames are committed.
        state = model.init_decoder_state(1, device=device, dtype=posterior_mu.dtype)
        prediction = None
        for replay_index in range(history_start, window_end):
            state, prediction = model.decode_step(
                posterior_mu[:, replay_index : replay_index + 1],
                local_root[:, replay_index : replay_index + 1],
                local_valid[:, replay_index : replay_index + 1],
                state,
            )
        if prediction is None:
            raise AssertionError("rolling replay did not decode the current token")

        # Offline decoding of the identical truncated window is an oracle for
        # the session-local cache implementation. Compare only the current
        # token because historical outputs are replay warm-up, not commits.
        window_offline = model.decode(
            posterior_mu[:, history_start:window_end],
            local_root[:, history_start:window_end],
            local_valid[:, history_start:window_end],
        )
        offline_current_continuous = window_offline.continuous_body[
            :, -FRAMES_PER_TOKEN:
        ]
        offline_current_contacts = window_offline.contact_logits[:, -FRAMES_PER_TOKEN:]
        window_max_abs = max(
            float((prediction.continuous_body - offline_current_continuous).abs().max()),
            float((prediction.contact_logits - offline_current_contacts).abs().max()),
        )
        cache_window_max_abs = max(cache_window_max_abs, window_max_abs)
        if window_max_abs > float(parity_atol):
            raise RuntimeError(
                f"offline/window-replay decoder parity failed for "
                f"{sample.dataset}/{sample.sample_id} at token {commit_index}: "
                f"max_abs={window_max_abs:.8g}, tolerance={parity_atol:.8g}"
            )
        continuous_chunks.append(prediction.continuous_body)
        contact_chunks.append(prediction.contact_logits)
        for name in trace_lists:
            trace_lists[name].append(window[name])
    streamed = BodyPrediction(
        continuous_body=torch.cat(continuous_chunks, dim=1),
        contact_logits=torch.cat(contact_chunks, dim=1),
    )
    rolling_reference_max_abs = max(
        float((streamed.continuous_body - reference_stream.continuous_body).abs().max()),
        float((streamed.contact_logits - reference_stream.contact_logits).abs().max()),
    )
    rolling_trace = {
        "timeline_position_ids": torch.cat(
            trace_lists["timeline_position_ids"], dim=0
        ).cpu(),
        "history_mask": torch.cat(trace_lists["history_mask"], dim=0).cpu(),
        "current_mask": torch.cat(trace_lists["current_mask"], dim=0).cpu(),
    }
    for name in (
        "window_origin", "history_start", "history_end", "window_end",
        "commit_token",
    ):
        rolling_trace[name] = torch.tensor(trace_lists[name], dtype=torch.long)
    rolling_trace["history_tokens"] = torch.tensor(int(history_tokens))
    rolling_trace["commit_tokens"] = torch.tensor(int(commit_tokens))
    expected_commits = torch.arange(posterior_mu.shape[1])
    if not torch.equal(rolling_trace["commit_token"], expected_commits):
        raise AssertionError("rolling scheduler skipped or duplicated a token")
    return ReconstructionResult(
        protocol=ROLLING_PROTOCOL,
        posterior_mu=posterior_mu.cpu(),
        local_root_motion=local_root.cpu(),
        local_root_valid_mask=local_valid.cpu(),
        streamed_body=BodyPrediction(
            streamed.continuous_body.cpu(), streamed.contact_logits.cpu()
        ),
        offline_body=BodyPrediction(
            offline.continuous_body.cpu(), offline.contact_logits.cpu()
        ),
        stream_offline_max_abs=cache_window_max_abs,
        reference_stream_body=BodyPrediction(
            reference_stream.continuous_body.cpu(),
            reference_stream.contact_logits.cpu(),
        ),
        rolling_reference_max_abs=rolling_reference_max_abs,
        rolling_trace=rolling_trace,
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    expanded = mask.expand_as(value)
    if not bool(expanded.any()):
        return 0.0
    return float(value[expanded].mean())


def reconstruction_metrics(
    sample: MotionSample,
    result: ReconstructionResult,
) -> dict[str, float | int | str]:
    target = sample.body_motion
    predicted = result.streamed_body.continuous_body[0]
    feature_valid = sample.body_feature_valid_mask
    position_error = (predicted[:, :BODY_POSITION_DIM] - target[:, :BODY_POSITION_DIM]).abs()
    velocity_start = BODY_POSITION_DIM + BODY_ROTATION_DIM
    velocity_error = (predicted[:, velocity_start:] - target[:, velocity_start:BODY_CONTINUOUS_DIM]).abs()

    pred_rotation = rotation_to_matrix(
        predicted[:, BODY_POSITION_DIM:velocity_start].reshape(-1, NUM_JOINTS, 6)
    )
    target_rotation = rotation_to_matrix(
        target[:, BODY_POSITION_DIM:velocity_start].reshape(-1, NUM_JOINTS, 6)
    )
    relative = pred_rotation.transpose(-1, -2) @ target_rotation
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    rotation_error = torch.rad2deg(torch.acos(cosine))
    rotation_valid = feature_valid[:, BODY_POSITION_DIM:velocity_start].reshape(
        -1, NUM_JOINTS, 6
    ).all(-1)

    target_contact = target[:, BODY_CONTINUOUS_DIM:].bool()
    predicted_contact = result.streamed_body.contact_logits[0].sigmoid() >= 0.5
    true_positive = int((predicted_contact & target_contact).sum())
    false_positive = int((predicted_contact & ~target_contact).sum())
    false_negative = int((~predicted_contact & target_contact).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    foot_indices = torch.tensor((7, 10, 8, 11))
    reconstructed_joints = recover_joint_positions(
        sample.root_motion, predicted
    )
    reconstructed_foot_positions = reconstructed_joints.index_select(
        1, foot_indices
    )
    position_foot_speed = predicted.new_zeros(target.shape[0], 4)
    position_foot_speed[1:] = (
        reconstructed_foot_positions[1:]
        - reconstructed_foot_positions[:-1]
    ).norm(dim=-1) * float(sample.fps)
    foot_position_valid = feature_valid[:, :BODY_POSITION_DIM].reshape(
        -1, NUM_JOINTS - 1, 3
    ).all(dim=-1).index_select(1, foot_indices - 1)
    transition_valid = torch.zeros_like(foot_position_valid)
    transition_valid[1:] = foot_position_valid[1:] & foot_position_valid[:-1]
    contact_valid = feature_valid[:, BODY_CONTINUOUS_DIM:]
    position_skating_valid = transition_valid & contact_valid

    predicted_velocity = predicted[:, velocity_start:].reshape(-1, NUM_JOINTS, 3)
    velocity_valid = feature_valid[:, velocity_start:BODY_CONTINUOUS_DIM].reshape(
        -1, NUM_JOINTS, 3
    ).all(dim=-1).index_select(1, foot_indices)
    contact_probability = result.streamed_body.contact_logits[0].sigmoid()
    predicted_velocity_speed = predicted_velocity.index_select(
        1, foot_indices
    ).norm(dim=-1)

    metrics = {
        "protocol": result.protocol,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "frames": int(target.shape[0]),
        "tokens": frame_count_to_token_count(target.shape[0]),
        "position_mae_m": _masked_mean(
            position_error, feature_valid[:, :BODY_POSITION_DIM]
        ),
        "velocity_mae_mps": _masked_mean(
            velocity_error, feature_valid[:, velocity_start:BODY_CONTINUOUS_DIM]
        ),
        "rotation_geodesic_deg": _masked_mean(rotation_error, rotation_valid),
        "contact_accuracy": float((predicted_contact == target_contact).float().mean()),
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "gt_contact_position_skating_mps": _masked_mean(
            target_contact.float() * position_foot_speed,
            position_skating_valid,
        ),
        "predicted_contact_position_skating_mps": _masked_mean(
            contact_probability * position_foot_speed,
            position_skating_valid,
        ),
        "gt_contact_velocity_feature_mps": _masked_mean(
            target_contact.float() * predicted_velocity_speed,
            velocity_valid & contact_valid,
        ),
        "stream_offline_max_abs": result.stream_offline_max_abs,
    }
    if result.reference_stream_body is not None:
        reference = result.reference_stream_body.continuous_body[0]
        reference_position_error = (
            predicted[:, :BODY_POSITION_DIM] - reference[:, :BODY_POSITION_DIM]
        ).abs()
        reference_velocity_error = (
            predicted[:, velocity_start:] - reference[:, velocity_start:]
        ).abs()
        reference_rotation = rotation_to_matrix(
            reference[:, BODY_POSITION_DIM:velocity_start].reshape(
                -1, NUM_JOINTS, 6
            )
        )
        rolling_relative = pred_rotation.transpose(-1, -2) @ reference_rotation
        rolling_cosine = (
            (rolling_relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
        ).clamp(-1.0, 1.0)
        rolling_rotation_error = torch.rad2deg(torch.acos(rolling_cosine))
        reference_contacts = (
            result.reference_stream_body.contact_logits[0].sigmoid() >= 0.5
        )
        metrics.update(
            {
                "rolling_stream_position_mae_m": _masked_mean(
                    reference_position_error,
                    feature_valid[:, :BODY_POSITION_DIM],
                ),
                "rolling_stream_velocity_mae_mps": _masked_mean(
                    reference_velocity_error,
                    feature_valid[:, velocity_start:BODY_CONTINUOUS_DIM],
                ),
                "rolling_stream_rotation_geodesic_deg": _masked_mean(
                    rolling_rotation_error, rotation_valid
                ),
                "rolling_stream_contact_disagreement": float(
                    (predicted_contact != reference_contacts).float().mean()
                ),
                "rolling_stream_max_abs": float(result.rolling_reference_max_abs),
                "cache_window_offline_max_abs": result.stream_offline_max_abs,
            }
        )
    if result.rolling_trace is not None:
        metrics.update(
            {
                "history_tokens": int(result.rolling_trace["history_tokens"]),
                "commit_tokens": int(result.rolling_trace["commit_tokens"]),
                "rolling_steps": int(result.rolling_trace["commit_token"].shape[0]),
            }
        )
    return metrics


def _output_paths(output_root: Path, dataset_name: str, sample_id: str) -> dict[str, Path]:
    root = output_root / dataset_name
    return {
        "original_video": root / "video" / "original" / f"{sample_id}.mp4",
        "reconstruction_video": root / "video" / "reconstruction" / f"{sample_id}.mp4",
        "original_motion": root / "motion" / "original" / f"{sample_id}.npz",
        "reconstruction_motion": root / "motion" / "reconstruction" / f"{sample_id}.npz",
        "metrics": root / "metrics" / f"{sample_id}.json",
    }


def _save_sample_outputs(
    sample: MotionSample,
    result: ReconstructionResult,
    metrics: Mapping[str, object],
    *,
    output_root: Path,
    dataset_name: str,
    render_video: bool,
    render_fps: int,
) -> dict[str, str]:
    paths = _output_paths(output_root, dataset_name, sample.sample_id)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    original_joints = recover_joint_positions(
        sample.root_motion, sample.body_motion
    )
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
        [reconstructed_continuous, (contact_logits.sigmoid() >= 0.5).float()], dim=-1
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
    _write_json(paths["metrics"], metrics)
    if render_video:
        render_joint_video(
            original_joints,
            paths["original_video"],
            fps=render_fps,
        )
        render_joint_video(
            reconstructed_joints,
            paths["reconstruction_video"],
            fps=render_fps,
        )
    return {name: str(path) for name, path in paths.items()}


def _mean_metrics(sample_metrics: list[Mapping[str, object]]) -> dict[str, float]:
    structural = {
        "frames", "tokens", "history_tokens", "commit_tokens", "rolling_steps",
    }
    numeric_keys = sorted(
        key
        for key, value in sample_metrics[0].items()
        if isinstance(value, (float, int)) and key not in structural
    )
    return {
        key: float(np.mean([float(metrics[key]) for metrics in sample_metrics]))
        for key in numeric_keys
    }


def evaluate_dataset(
    model: BodyVAE,
    *,
    dataset_name: str,
    dataset_config=None,
    dataset=None,
    sample_count: int,
    output_root: Path,
    device: torch.device,
    parity_atol: float,
    render_video: bool,
    render_fps: int,
    mode: str,
    window_config=None,
) -> dict[str, object]:
    if dataset is None:
        if dataset_name not in DATASET_TYPES:
            raise ValueError(f"unsupported VAE evaluation dataset {dataset_name!r}")
        if dataset_config is None:
            raise ValueError("dataset_config is required when dataset is not provided")
        dataset = DATASET_TYPES[dataset_name](
            meta_paths=[dataset_config.val_meta_path],
            split="val",
            artifact_path=dataset_config.artifact_path,
            text_path=dataset_config.get("text_path"),
            fps=model.fps,
        )
    if len(dataset) < sample_count:
        raise RuntimeError(
            f"{dataset_name} val split contains only {len(dataset)} samples, "
            f"expected {sample_count}"
        )
    manifest_samples = []
    all_metrics = []
    for index in range(sample_count):
        sample = load_motion_sample(dataset[index], expected_fps=model.fps)
        if mode == "stream":
            result = stream_reconstruct(
                model, sample, device=device, parity_atol=parity_atol
            )
        elif mode == "rolling":
            if window_config is None:
                raise ValueError("rolling evaluation requires window configuration")
            result = rolling_reconstruct(
                model,
                sample,
                device=device,
                history_tokens=int(window_config.history_tokens),
                commit_tokens=int(window_config.commit_tokens),
                parity_atol=parity_atol,
            )
        else:
            raise ValueError(f"unsupported VAE reconstruction mode {mode!r}")
        metrics = reconstruction_metrics(sample, result)
        outputs = _save_sample_outputs(
            sample,
            result,
            metrics,
            output_root=output_root,
            dataset_name=dataset_name,
            render_video=render_video,
            render_fps=render_fps,
        )
        all_metrics.append(metrics)
        manifest_samples.append(
            {
                "index": index,
                "sample_id": sample.sample_id,
                "source_dataset": sample.dataset,
                "frames": int(sample.body_motion.shape[0]),
                "outputs": outputs,
            }
        )
        print(
            f"[{dataset_name} {index + 1}/{sample_count}] {sample.sample_id}: "
            f"position={metrics['position_mae_m']:.6f}m, "
            f"rotation={metrics['rotation_geodesic_deg']:.4f}deg"
        )
    dataset_output = output_root / dataset_name
    summary = {
        "dataset": dataset_name,
        "protocol": all_metrics[0]["protocol"],
        "sample_count": sample_count,
        "mean_metrics": _mean_metrics(all_metrics),
    }
    _write_json(dataset_output / "manifest.json", {"samples": manifest_samples})
    _write_json(dataset_output / "summary.json", summary)
    return summary


def _load_model(cfg, device: torch.device) -> tuple[BodyVAE, dict[str, object]]:
    model = BodyVAE(
        **OmegaConf.to_container(cfg.model.params, resolve=True),
        motion_stats_path=str(cfg.model.motion_stats_path),
    )
    load_vae_checkpoint(model, cfg.model.checkpoint_path)
    return model.to(device), {"path": str(cfg.model.checkpoint_path), "weights": "ema"}


def run(cfg, *, mode: str) -> dict[str, object]:
    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA evaluation requested but unavailable: {device}")
    if device.type == "cuda":
        # Full-sequence and single-token Conv1d otherwise select different TF32
        # kernels on Ada GPUs, inflating a purely numerical parity difference to
        # ~4e-3 in physical channels. The runtime path itself is stream-only;
        # this evaluation disables TF32 so its offline oracle remains meaningful.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    sample_count = int(cfg.sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    output_root = Path(str(cfg.output_dir))
    model, checkpoint_metadata = _load_model(cfg, device)
    dataset_summaries = {}
    for dataset_name, dataset_config in cfg.datasets.items():
        dataset_summaries[dataset_name] = evaluate_dataset(
            model,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            sample_count=sample_count,
            output_root=output_root,
            device=device,
            parity_atol=float(cfg.stream_parity_atol),
            render_video=bool(cfg.render.enabled),
            render_fps=int(cfg.render.fps),
            mode=mode,
            window_config=cfg.get("window"),
        )
    protocol = STREAM_PROTOCOL if mode == "stream" else ROLLING_PROTOCOL
    summary = {
        "protocol": protocol,
        "mode": mode,
        "root_policy": "source explicit root shared by original and reconstruction",
        "latent_policy": "deterministic raw posterior mu, streamed after LDF unnormalization boundary",
        "checkpoint": checkpoint_metadata,
        "datasets": dataset_summaries,
    }
    if mode == "rolling":
        summary["window"] = OmegaConf.to_container(cfg.window, resolve=True)
    _write_json(output_root / "summary.json", summary)
    return summary


def load_task_config(default_config: str):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.device:
        cfg.device = args.device
    if args.output:
        cfg.output_dir = args.output
    if args.sample_count is not None:
        cfg.sample_count = args.sample_count
    if args.skip_video:
        cfg.render.enabled = False
    return cfg
