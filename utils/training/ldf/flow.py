"""Flow-matching algebra for fixed-noise LDF training rollouts."""

from __future__ import annotations

import torch

from utils.conditions.ldf import HybridMotion


def build_span_beta(
    *,
    span_tokens: int,
    initial_history_tokens: int,
    active_tokens: int,
    phase_offset: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Return history/active/frontier beta from one continuous formula."""

    span_tokens = int(span_tokens)
    history_end = int(initial_history_tokens) + int(step_index)
    active_tokens = int(active_tokens)
    if span_tokens <= 0 or active_tokens <= 0:
        raise ValueError("span_tokens and active_tokens must be positive")
    if history_end < 0 or history_end + active_tokens > span_tokens:
        raise ValueError("active band does not fit inside the source span")
    if not torch.is_tensor(phase_offset) or phase_offset.ndim != 1:
        raise TypeError("phase_offset must be a floating [B] tensor")
    if not phase_offset.is_floating_point():
        raise TypeError("phase_offset must be floating point")
    if bool((phase_offset < 0).any()) or bool(
        (phase_offset >= 1.0 / active_tokens).any()
    ):
        raise ValueError("phase_offset must lie in [0,1/active_tokens)")

    positions = torch.arange(
        span_tokens,
        device=phase_offset.device,
        dtype=phase_offset.dtype,
    )
    return torch.clamp(
        (positions[None] - float(history_end) + 1.0) / float(active_tokens)
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


def flow_velocity_target(
    clean_motion: HybridMotion,
    noise: HybridMotion,
) -> HybridMotion:
    """Return the normalized flow target v*=x0-epsilon."""

    clean_motion.validate()
    noise.validate()
    return HybridMotion(
        clean_motion.root_motion - noise.root_motion,
        clean_motion.latent_motion - noise.latent_motion,
    )


def recover_clean_for_self_forcing(
    noisy_value: torch.Tensor,
    beta: torch.Tensor,
    predicted_velocity: torch.Tensor,
) -> torch.Tensor:
    """Low-error clean estimate used only at a self-forcing boundary."""

    if tuple(noisy_value.shape) != tuple(predicted_velocity.shape):
        raise ValueError("noisy_value and predicted_velocity must share shape")
    while beta.ndim < noisy_value.ndim:
        beta = beta.unsqueeze(-1)
    return noisy_value + beta.to(noisy_value) * predicted_velocity


def recover_clean_for_full_gradient_auxiliary(
    predicted_velocity: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Full-gradient clean estimate reserved for future auxiliary losses."""

    if tuple(predicted_velocity.shape) != tuple(noise.shape):
        raise ValueError("predicted_velocity and noise must share shape")
    return predicted_velocity + noise


__all__ = [
    "build_span_beta",
    "flow_velocity_target",
    "mix_fixed_noise",
    "recover_clean_for_full_gradient_auxiliary",
    "recover_clean_for_self_forcing",
]
