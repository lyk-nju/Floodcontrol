"""Body VAE computation interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.tools.wan_vae_1d import CausalBodyVAE
from utils.conditions.vae import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    BodyPrediction,
    VAEDecoderState,
    VAEInput,
    VAEPrediction,
    VAEPosterior,
)
from utils.motion_process import recover_local_root
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
    require_aligned_frame_count,
)


def _load_statistic(data, name: str, expected_dim: int) -> torch.Tensor:
    if name not in data:
        raise ValueError(f"statistics file is missing {name!r}")
    value = torch.from_numpy(np.asarray(data[name])).float()
    if tuple(value.shape) != (expected_dim,):
        raise ValueError(f"{name} must have shape [{expected_dim}]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    if name.endswith("std") and bool((value <= 0).any()):
        raise ValueError(f"{name} must be positive")
    return value


def _load_motion_statistics(path: str | Path) -> tuple[torch.Tensor, ...]:
    with np.load(path, allow_pickle=False) as data:
        return tuple(
            _load_statistic(data, name, dim)
            for name, dim in (
                ("body_cont_mean", BODY_CONTINUOUS_DIM),
                ("body_cont_std", BODY_CONTINUOUS_DIM),
                ("local_root_mean", 4),
                ("local_root_std", 4),
            )
        )


def _load_latent_statistics(
    path: str | Path, latent_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        mean_key = "mean" if "mean" in data else "latent_mu_mean"
        std_key = "std" if "std" in data else "latent_mu_std"
        mean = _load_statistic(data, mean_key, latent_dim)
        std = _load_statistic(data, std_key, latent_dim)
    return mean, std


class BodyVAE(nn.Module):
    """Body-only VAE with explicit physical and latent-space boundaries.

    ``encode``/``decode`` operate on raw posterior latents. ``tokenize`` and
    ``detokenize`` are the normalized deployment interface used by the LDF.
    Statistics are loaded from two ordinary NPZ files. Decoder history is an
    explicit value passed by the caller; the module owns no hidden cache.
    """

    def __init__(
        self,
        *,
        motion_stats_path: str | Path,
        latent_stats_path: str | Path | None = None,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
        fps: float = 20.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.fps = float(fps)
        body_mean, body_std, local_mean, local_std = _load_motion_statistics(
            motion_stats_path
        )

        self.register_buffer(
            "body_cont_mean", body_mean
        )
        self.register_buffer(
            "body_cont_std", body_std
        )
        self.register_buffer(
            "local_root_mean", local_mean
        )
        self.register_buffer(
            "local_root_std", local_std
        )
        latent_mean, latent_std = (
            _load_latent_statistics(latent_stats_path, self.latent_dim)
            if latent_stats_path is not None
            else (torch.empty(0), torch.empty(0))
        )
        # Latent statistics are computed after training and never belong to a
        # training checkpoint.
        self.register_buffer("latent_mean", latent_mean, persistent=False)
        self.register_buffer("latent_std", latent_std, persistent=False)

        self.model = CausalBodyVAE(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    @property
    def encoder_context_tokens(self) -> int:
        return self.model.encoder_context_tokens

    @property
    def decoder_context_tokens(self) -> int:
        return self.model.decoder_context_tokens

    @property
    def latent_statistics_ready(self) -> bool:
        return self.latent_mean.numel() == self.latent_dim

    def normalize_body(self, body_motion: torch.Tensor) -> torch.Tensor:
        if body_motion.shape[-1] != BODY_DIM:
            raise ValueError("body_motion must end in 265")
        continuous = (
            body_motion[..., :BODY_CONTINUOUS_DIM] - self.body_cont_mean
        ) / self.body_cont_std
        contacts = body_motion[..., BODY_CONTINUOUS_DIM:]
        return torch.cat([continuous, contacts], dim=-1)

    def normalize_local_root(self, local_root_motion: torch.Tensor) -> torch.Tensor:
        return (local_root_motion - self.local_root_mean) / self.local_root_std

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.latent_statistics_ready:
            raise RuntimeError("latent mu statistics must be loaded before tokenization")
        normalized = (latent - self.latent_mean) / self.latent_std
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("normalized latent contains non-finite values")
        return normalized

    def unnormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.latent_statistics_ready:
            raise RuntimeError("latent mu statistics must be loaded before detokenization")
        raw = latent * self.latent_std + self.latent_mean
        if not bool(torch.isfinite(raw).all()):
            raise ValueError("unnormalized latent contains non-finite values")
        return raw

    def encode(
        self, body_motion: torch.Tensor, frame_valid_mask: torch.Tensor
    ) -> VAEPosterior:
        """Encode physical body motion into a raw posterior."""

        if body_motion.ndim != 3 or body_motion.shape[-1] != BODY_DIM:
            raise ValueError("body_motion must be [B,F,265]")
        if (
            tuple(frame_valid_mask.shape) != tuple(body_motion.shape[:2])
            or frame_valid_mask.dtype != torch.bool
        ):
            raise ValueError("frame_valid_mask must be bool [B,F]")
        normalized = self.normalize_body(body_motion)
        normalized = torch.where(
            frame_valid_mask[..., None], normalized, torch.zeros_like(normalized)
        )
        posterior = self.model.encode(normalized)
        if not bool(torch.isfinite(posterior.mu).all()):
            raise ValueError("VAE posterior mu contains non-finite values")
        if not bool(torch.isfinite(posterior.logvar).all()):
            raise ValueError("VAE posterior logvar contains non-finite values")
        return posterior

    def encode_window(
        self,
        body_with_context: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        context_token_count: int,
    ) -> VAEPosterior:
        """Encode raw posterior values for an active window after warm-up context."""

        context_token_count = int(context_token_count)
        if not 0 <= context_token_count <= self.encoder_context_tokens:
            raise ValueError("context_token_count exceeds the encoder causal context")
        frames = require_aligned_frame_count(body_with_context.shape[1])
        total_tokens = frame_count_to_token_count(frames)
        if context_token_count >= total_tokens:
            raise ValueError("encode_window requires at least one active token")
        posterior = self.encode(body_with_context, frame_valid_mask)
        return VAEPosterior(
            mu=posterior.mu[:, context_token_count:],
            logvar=posterior.logvar[:, context_token_count:],
        )

    def decode(
        self,
        latent_tokens: torch.Tensor,
        local_root_motion: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        """Decode raw latent tokens into physical body motion."""

        output = self.model.decode(
            latent_tokens,
            self.normalize_local_root(local_root_motion),
            local_root_valid_mask,
        )
        return self._physical_prediction(output, frame_valid_mask)

    def _physical_prediction(
        self,
        output: BodyPrediction,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        physical = output.continuous_body * self.body_cont_std + self.body_cont_mean
        contact_logits = output.contact_logits
        if frame_valid_mask is not None:
            if tuple(frame_valid_mask.shape) != tuple(physical.shape[:2]):
                raise ValueError("frame_valid_mask does not match decoded frames")
            physical = torch.where(
                frame_valid_mask[..., None], physical, torch.zeros_like(physical)
            )
            contact_logits = torch.where(
                frame_valid_mask[..., None],
                contact_logits,
                contact_logits.new_full((), -20.0),
            )
        result = BodyPrediction(physical, contact_logits)
        result.validate()
        return result

    def forward(self, inputs: VAEInput) -> VAEPrediction:
        if not isinstance(inputs, VAEInput):
            raise TypeError("BodyVAE.forward requires VAEInput")
        inputs.validate()
        posterior = self.encode(inputs.body_motion, inputs.frame_valid_mask)
        latent = posterior.sample()
        local_root, local_valid = recover_local_root(
            inputs.root_motion,
            inputs.previous_root_frame,
            fps=self.fps,
            previous_root_valid_mask=inputs.previous_root_valid_mask,
        )
        frame_valid = inputs.frame_valid_mask.reshape(
            inputs.frame_valid_mask.shape[0], -1, FRAMES_PER_TOKEN, 1
        )
        local_valid = local_valid & frame_valid
        local_root = torch.where(local_valid, local_root, torch.zeros_like(local_root))
        body = self.decode(latent, local_root, local_valid, inputs.frame_valid_mask)
        return VAEPrediction(body, posterior, latent, local_root, local_valid)

    @torch.no_grad()
    def tokenize(
        self, body_motion: torch.Tensor, frame_valid_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.normalize_latent(self.encode(body_motion, frame_valid_mask).mu)

    @torch.no_grad()
    def tokenize_window(
        self,
        body_with_context: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        context_token_count: int,
    ) -> torch.Tensor:
        posterior = self.encode_window(
            body_with_context, frame_valid_mask, context_token_count
        )
        return self.normalize_latent(posterior.mu)

    @torch.no_grad()
    def detokenize(
        self,
        normalized_mu: torch.Tensor,
        local_root_motion: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        return self.decode(
            self.unnormalize_latent(normalized_mu),
            local_root_motion,
            local_root_valid_mask,
            frame_valid_mask,
        )

    def init_decoder_state(
        self, batch_size: int, *, device=None, dtype=torch.float32
    ) -> VAEDecoderState:
        device = torch.device(device or self.body_cont_mean.device)
        return VAEDecoderState(
            self.model.init_decoder_cache(batch_size, device=device, dtype=dtype)
        )

    def decode_step(
        self,
        latent_token: torch.Tensor,
        local_root_patch: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        state: VAEDecoderState,
    ) -> tuple[VAEDecoderState, BodyPrediction]:
        """Decode one raw latent token with explicit causal caches."""

        next_caches, output = self.model.decode_step(
            latent_token,
            self.normalize_local_root(local_root_patch),
            local_root_valid_mask,
            state.caches,
        )
        return VAEDecoderState(next_caches), self._physical_prediction(output)

    @torch.no_grad()
    def detokenize_step(
        self,
        normalized_token: torch.Tensor,
        local_root_patch: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        state: VAEDecoderState,
    ) -> tuple[VAEDecoderState, BodyPrediction]:
        """Decode one normalized tokenizer token with explicit causal caches."""

        return self.decode_step(
            self.unnormalize_latent(normalized_token),
            local_root_patch,
            local_root_valid_mask,
            state,
        )


__all__ = ["BodyVAE"]
