"""Deterministic stream and finite-history BodyVAE reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import BodyPrediction
from utils.motion_process import recover_local_root
from utils.token_frame import FRAMES_PER_TOKEN


STREAM_PROTOCOL = "deterministic-mu-stream-decode-v1"
ROLLING_PROTOCOL = "deterministic-mu-history-replay-v1"


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


def load_motion_sample(
    sample: Mapping[str, object], *, expected_fps: float
) -> MotionSample:
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
    """Decode deterministic posterior means with one persistent decoder state."""

    device = torch.device(device)
    body = sample.body_motion[None].to(device)
    body_feature_valid = sample.body_feature_valid_mask[None].to(device)
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
    posterior_mu = model.encode(
        body,
        frame_valid,
        body_feature_valid_mask=body_feature_valid,
    ).mu
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
    """Create a right-aligned retained-history view and one current token."""

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
    body_feature_valid = sample.body_feature_valid_mask[None].to(device)
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
    posterior_mu = model.encode(
        body,
        frame_valid,
        body_feature_valid_mask=body_feature_valid,
    ).mu
    local_root, local_valid = recover_local_root(
        root,
        previous_root,
        fps=sample.fps,
        previous_root_valid_mask=previous_valid,
    )
    offline = model.decode(posterior_mu, local_root, local_valid, frame_valid)

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
        "window_origin",
        "history_start",
        "history_end",
        "window_end",
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


__all__ = [
    "ROLLING_PROTOCOL",
    "STREAM_PROTOCOL",
    "MotionSample",
    "ReconstructionResult",
    "create_rolling_window",
    "load_motion_sample",
    "rolling_reconstruct",
    "stream_reconstruct",
]
