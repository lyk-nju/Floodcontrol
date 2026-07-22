"""Flow-matching algebra for fixed-noise LDF training rollouts."""

from __future__ import annotations

import torch

from utils.conditions.ldf import HybridMotion


def build_span_beta(
    *,
    span_tokens: int,
    initial_history_tokens: torch.Tensor,
    active_tokens: int,
    phase_offset: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Return per-sample history/active/frontier beta from one formula."""

    span_tokens = int(span_tokens)
    active_tokens = int(active_tokens)
    if span_tokens <= 0 or active_tokens <= 0:
        raise ValueError("span_tokens and active_tokens must be positive")
    if not torch.is_tensor(phase_offset) or phase_offset.ndim != 1:
        raise TypeError("phase_offset must be a floating [B] tensor")
    if not phase_offset.is_floating_point():
        raise TypeError("phase_offset must be floating point")
    history = torch.as_tensor(
        initial_history_tokens,
        device=phase_offset.device,
        dtype=torch.long,
    ).reshape(-1)
    if tuple(history.shape) != tuple(phase_offset.shape):
        raise ValueError("initial_history_tokens and phase_offset must both be [B]")
    history_end = history + int(step_index)

    positions = torch.arange(
        span_tokens,
        device=phase_offset.device,
        dtype=phase_offset.dtype,
    )
    return torch.clamp(
        (positions[None] - history_end[:, None].to(phase_offset.dtype) + 1.0)
        / float(active_tokens)
        - phase_offset[:, None],
        min=0.0,
        max=1.0,
    )


def mix_fixed_noise(
    clean_motion: HybridMotion,
    noise: HybridMotion,
    beta: torch.Tensor,
) -> HybridMotion:
    """Construct x_beta while keeping noise fixed on the absolute token axis."""

    clean_motion.validate()
    noise.validate()
    if tuple(clean_motion.root_motion.shape) != tuple(noise.root_motion.shape) or tuple(
        clean_motion.latent_motion.shape
    ) != tuple(noise.latent_motion.shape):
        raise ValueError("clean motion and noise must have identical shapes")
    if tuple(beta.shape) != tuple(clean_motion.root_motion.shape[:2]):
        raise ValueError("beta must match the HybridMotion [B,T] axis")
    beta = beta.to(clean_motion.root_motion)
    return HybridMotion(
        (1.0 - beta[..., None, None]) * clean_motion.root_motion
        + beta[..., None, None] * noise.root_motion,
        (1.0 - beta[..., None]) * clean_motion.latent_motion
        + beta[..., None] * noise.latent_motion,
    )


__all__ = [
    "build_span_beta",
    "mix_fixed_noise",
]
