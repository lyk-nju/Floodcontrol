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

from datasets.babel import load_babel_records
from datasets.humanml3d import load_humanml3d_records
from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    CONTRACT_VERSION,
    FRAMES_PER_TOKEN,
    NUM_JOINTS,
    ROOT_DIM,
    BodyPrediction,
)
from utils.motion_representation import (
    MOTION_CONVERTER_VERSION,
    derive_patched_local_root,
    rotation_6d_to_matrix,
)
from utils.training.vae.checkpoint import load_ema_checkpoint
from utils.visualization.skeleton import (
    get_humanml3d_chains,
    render_simple_skeleton_video,
)


STREAM_PROTOCOL = "deterministic-mu-stream-decode-v1"
ROLLING_PROTOCOL = "deterministic-mu-history-replay-v1"
DATASET_LOADERS = {
    "humanml3d": load_humanml3d_records,
    "babel": load_babel_records,
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


def _read_scalar(data, name: str, path: Path):
    if name not in data:
        raise ValueError(f"motion artifact is missing {name!r} in {path}")
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"motion artifact {name!r} must be scalar in {path}")
    return value.item()


def load_motion_sample(record: Mapping[str, object], *, expected_fps: float) -> MotionSample:
    path = Path(record["artifact"])
    with np.load(path, allow_pickle=False) as data:
        if str(_read_scalar(data, "contract_version", path)) != CONTRACT_VERSION:
            raise ValueError(f"motion artifact contract version mismatch in {path}")
        if str(_read_scalar(data, "converter_version", path)) != MOTION_CONVERTER_VERSION:
            raise ValueError(f"motion artifact converter version mismatch in {path}")
        fps = float(_read_scalar(data, "fps", path))
        root = torch.from_numpy(data["root_motion"]).float()
        body = torch.from_numpy(data["body_motion"]).float()
        feature_valid = torch.from_numpy(data["body_feature_valid_mask"]).bool()
        previous_root = (
            torch.from_numpy(data["previous_root_frame"]).float()
            if "previous_root_frame" in data
            else None
        )
    if not np.isclose(fps, expected_fps, rtol=0.0, atol=1e-6):
        raise ValueError(
            f"motion artifact FPS mismatch in {path}: expected {expected_fps}, got {fps}"
        )
    if root.ndim != 2 or tuple(root.shape[1:]) != (ROOT_DIM,):
        raise ValueError(f"root_motion must be [F,{ROOT_DIM}] in {path}")
    if body.ndim != 2 or tuple(body.shape[1:]) != (BODY_DIM,):
        raise ValueError(f"body_motion must be [F,{BODY_DIM}] in {path}")
    if root.shape[0] != body.shape[0] or root.shape[0] % FRAMES_PER_TOKEN:
        raise ValueError(f"motion artifact frame contract mismatch in {path}")
    if tuple(feature_valid.shape) != tuple(body.shape):
        raise ValueError(f"body_feature_valid_mask must match body_motion in {path}")
    if previous_root is not None and tuple(previous_root.shape) != (ROOT_DIM,):
        raise ValueError(f"previous_root_frame must be [{ROOT_DIM}] in {path}")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (root, body, previous_root)
        if value is not None
    ):
        raise ValueError(f"motion artifact contains non-finite values in {path}")
    return MotionSample(
        sample_id=str(record["name"]),
        dataset=str(record["dataset"]),
        root_motion=root,
        body_motion=body,
        body_feature_valid_mask=feature_valid,
        previous_root_frame=previous_root,
        fps=fps,
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
    local_root, local_valid = derive_patched_local_root(
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
        state, prediction = model.stream_decode_step(
            posterior_mu[:, token_index : token_index + 1],
            local_root[:, token_index : token_index + 1],
            state,
            local_root_valid_mask=local_valid[:, token_index : token_index + 1],
            normalized_latent=False,
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
    if state.token_index != posterior_mu.shape[1]:
        raise AssertionError("decoder state token index does not match committed tokens")
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
    local_root, local_valid = derive_patched_local_root(
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
        reference_state, reference_prediction = model.stream_decode_step(
            posterior_mu[:, token_index : token_index + 1],
            local_root[:, token_index : token_index + 1],
            reference_state,
            local_root_valid_mask=local_valid[:, token_index : token_index + 1],
            normalized_latent=False,
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
            state, prediction = model.stream_decode_step(
                posterior_mu[:, replay_index : replay_index + 1],
                local_root[:, replay_index : replay_index + 1],
                state,
                local_root_valid_mask=local_valid[:, replay_index : replay_index + 1],
                normalized_latent=False,
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


def body_to_global_joints(root_motion: torch.Tensor, body_motion: torch.Tensor) -> torch.Tensor:
    if root_motion.ndim != 2 or tuple(root_motion.shape[1:]) != (ROOT_DIM,):
        raise ValueError("root_motion must be [F,5]")
    if body_motion.ndim != 2 or body_motion.shape[1] < BODY_POSITION_DIM:
        raise ValueError("body_motion must contain non-root joint positions")
    if root_motion.shape[0] != body_motion.shape[0]:
        raise ValueError("root_motion and body_motion must share frame length")
    non_root = body_motion[:, :BODY_POSITION_DIM].reshape(-1, NUM_JOINTS - 1, 3).clone()
    non_root[..., 0] += root_motion[:, None, 0]
    non_root[..., 2] += root_motion[:, None, 2]
    return torch.cat([root_motion[:, None, :3], non_root], dim=1)


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

    pred_rotation = rotation_6d_to_matrix(
        predicted[:, BODY_POSITION_DIM:velocity_start].reshape(-1, NUM_JOINTS, 6)
    )
    target_rotation = rotation_6d_to_matrix(
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
    predicted_velocity = predicted[:, velocity_start:].reshape(-1, NUM_JOINTS, 3)
    target_velocity = target[:, velocity_start:BODY_CONTINUOUS_DIM].reshape(
        -1, NUM_JOINTS, 3
    )
    contact_probability = result.streamed_body.contact_logits[0].sigmoid()
    reconstructed_skating = (
        contact_probability
        * predicted_velocity.index_select(1, foot_indices).norm(dim=-1)
    ).mean()
    original_skating = (
        target_contact.float()
        * target_velocity.index_select(1, foot_indices).norm(dim=-1)
    ).mean()

    metrics = {
        "protocol": result.protocol,
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "frames": int(target.shape[0]),
        "tokens": int(target.shape[0] // FRAMES_PER_TOKEN),
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
        "original_skating_mps": float(original_skating),
        "reconstruction_skating_mps": float(reconstructed_skating),
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
        reference_rotation = rotation_6d_to_matrix(
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

    original_joints = body_to_global_joints(sample.root_motion, sample.body_motion)
    reconstructed_continuous = result.streamed_body.continuous_body[0]
    reconstructed_joints = body_to_global_joints(
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
        chains = get_humanml3d_chains()
        render_simple_skeleton_video(
            original_joints.numpy(), chains, str(paths["original_video"]), fps=render_fps
        )
        render_simple_skeleton_video(
            reconstructed_joints.numpy(),
            chains,
            str(paths["reconstruction_video"]),
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
    dataset_config,
    sample_count: int,
    output_root: Path,
    device: torch.device,
    parity_atol: float,
    render_video: bool,
    render_fps: int,
    mode: str,
    window_config=None,
) -> dict[str, object]:
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"unsupported VAE evaluation dataset {dataset_name!r}")
    records = DATASET_LOADERS[dataset_name](
        [dataset_config.val_meta_path], artifact_path=dataset_config.artifact_path
    )[:sample_count]
    if len(records) != sample_count:
        raise RuntimeError(
            f"{dataset_name} val split contains only {len(records)} samples, "
            f"expected {sample_count}"
        )
    manifest_samples = []
    all_metrics = []
    for index, record in enumerate(records):
        sample = load_motion_sample(record, expected_fps=model.fps)
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
                "source_artifact": str(record["artifact"]),
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
        require_latent_statistics=False,
    )
    checkpoint_metadata = load_ema_checkpoint(model, cfg.model.checkpoint_path)
    return model.eval().to(device), checkpoint_metadata


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
