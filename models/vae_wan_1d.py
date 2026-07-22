"""Body VAE computation interface."""

from __future__ import annotations

from collections.abc import Mapping
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
    MOTION_FPS,
    frame_count_to_token_count,
    frame_valid_to_token_valid,
    prefix_valid_token_count,
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


def _load_motion_statistics(path: str | Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        return {
            name: _load_statistic(data, name, dim)
            for name, dim in (
                ("body_cont_mean", BODY_CONTINUOUS_DIM),
                ("body_cont_std", BODY_CONTINUOUS_DIM),
                ("local_root_mean", 4),
                ("local_root_std", 4),
            )
        }


def _validate_motion_statistics(
    statistics: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    validated: dict[str, torch.Tensor] = {}
    for name, dim in (
        ("body_cont_mean", BODY_CONTINUOUS_DIM),
        ("body_cont_std", BODY_CONTINUOUS_DIM),
        ("local_root_mean", 4),
        ("local_root_std", 4),
    ):
        if name not in statistics:
            raise ValueError(f"motion statistics are missing {name!r}")
        value = torch.as_tensor(statistics[name], dtype=torch.float32).detach().clone()
        if tuple(value.shape) != (dim,):
            raise ValueError(f"{name} must have shape [{dim}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
        if name.endswith("std") and bool((value <= 0).any()):
            raise ValueError(f"{name} must be positive")
        validated[name] = value
    return validated


class BodyVAE(nn.Module):
    """Body-only VAE with explicit physical and raw-latent boundaries.

    ``encode``/``tokenize`` return raw posterior latents and
    ``decode``/``detokenize`` consume raw latents.  Physical body/local-root
    statistics may come from the VAE training NPZ or directly from a checkpoint
    loader. Decoder history is explicit; the module owns no hidden cache.
    """

    def __init__(
        self,
        *,
        motion_stats_path: str | Path | None = None,
        motion_statistics: Mapping[str, torch.Tensor] | None = None,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.fps = MOTION_FPS
        if (motion_stats_path is None) == (motion_statistics is None):
            raise ValueError(
                "provide exactly one of motion_stats_path or motion_statistics"
            )
        statistics = (
            _load_motion_statistics(motion_stats_path)
            if motion_stats_path is not None
            else _validate_motion_statistics(motion_statistics)
        )
        for name, value in statistics.items():
            self.register_buffer(name, value)

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

    def normalize_body(self, body_motion: torch.Tensor) -> torch.Tensor:
        if body_motion.shape[-1] != BODY_DIM:
            raise ValueError(f"body_motion must end in {BODY_DIM}")
        continuous = (
            body_motion[..., :BODY_CONTINUOUS_DIM] - self.body_cont_mean
        ) / self.body_cont_std
        contacts = body_motion[..., BODY_CONTINUOUS_DIM:]
        return torch.cat([continuous, contacts], dim=-1)

    def normalize_local_root(self, local_root_motion: torch.Tensor) -> torch.Tensor:
        return (local_root_motion - self.local_root_mean) / self.local_root_std

    def encode(
        self, body_motion: torch.Tensor, frame_valid_mask: torch.Tensor
    ) -> VAEPosterior:
        """Encode physical body motion into a raw posterior."""

        if body_motion.ndim != 3 or body_motion.shape[-1] != BODY_DIM:
            raise ValueError(f"body_motion must be [B,F,{BODY_DIM}]")
        if (
            tuple(frame_valid_mask.shape) != tuple(body_motion.shape[:2])
            or frame_valid_mask.dtype != torch.bool
        ):
            raise ValueError("frame_valid_mask must be bool [B,F]")
        if not bool(torch.isfinite(body_motion[frame_valid_mask]).all()):
            raise ValueError("valid body_motion frames contain non-finite values")
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

    def decode(
        self,
        latent_tokens: torch.Tensor,
        local_root_motion: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        """Decode raw latent tokens into physical body motion."""

        if not bool(torch.isfinite(latent_tokens).all()):
            raise ValueError("latent_tokens contain non-finite values")
        if not bool(torch.isfinite(local_root_motion).all()):
            raise ValueError("local_root_motion contains non-finite values")

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
        if not bool(torch.isfinite(physical).all()) or not bool(
            torch.isfinite(contact_logits).all()
        ):
            raise ValueError("VAE decoder produced non-finite values")
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
        return self.encode(body_motion, frame_valid_mask).mu

    @torch.no_grad()
    def tokenize_window(
        self,
        body_with_context: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        context_token_count: torch.Tensor,
    ) -> torch.Tensor:
        """Encode active motion as raw deterministic posterior means.

        ``body_with_context`` is laid out as a valid left prefix followed by
        right padding. ``context_token_count`` locates the first active token
        independently for each sample.
        """

        if body_with_context.ndim != 3 or body_with_context.shape[-1] != BODY_DIM:
            raise ValueError(f"body_with_context must be [B,F,{BODY_DIM}]")
        if not torch.is_tensor(context_token_count):
            raise TypeError("context_token_count must be a tensor")
        if context_token_count.dtype != torch.long:
            raise TypeError("context_token_count must be long [B]")
        batch = body_with_context.shape[0]
        if tuple(context_token_count.shape) != (batch,):
            raise ValueError("context_token_count must be long [B]")
        frames = require_aligned_frame_count(body_with_context.shape[1])
        total_tokens = frame_count_to_token_count(frames)
        if tuple(frame_valid_mask.shape) != tuple(body_with_context.shape[:2]):
            raise ValueError("frame_valid_mask must match body_with_context [B,F]")
        if frame_valid_mask.dtype != torch.bool:
            raise TypeError("frame_valid_mask must be bool [B,F]")

        frame_patches = frame_valid_mask.reshape(
            batch, total_tokens, FRAMES_PER_TOKEN
        )
        if not bool((frame_patches == frame_patches[..., :1]).all()):
            raise ValueError(
                "frame validity must be constant within each four-frame token"
            )
        token_valid = frame_valid_to_token_valid(frame_valid_mask)
        valid_token_count = prefix_valid_token_count(token_valid)
        context_token_count = context_token_count.to(device=body_with_context.device)
        if bool((context_token_count < 0).any()) or bool(
            (context_token_count > self.encoder_context_tokens).any()
        ):
            raise ValueError("context_token_count exceeds the encoder causal context")
        if bool((context_token_count >= valid_token_count).any()):
            raise ValueError("tokenize_window requires at least one active token per sample")

        active_token_count = valid_token_count - context_token_count
        max_active_tokens = int(active_token_count.max().item())
        active_offsets = torch.arange(
            max_active_tokens, device=body_with_context.device
        )
        gather_index = context_token_count[:, None] + active_offsets[None]
        active_valid = active_offsets[None] < active_token_count[:, None]
        safe_index = gather_index.clamp(max=total_tokens - 1)

        posterior = self.encode(body_with_context, frame_valid_mask)
        latent_index = safe_index[..., None].expand(
            batch, max_active_tokens, posterior.mu.shape[-1]
        )
        raw_mu = posterior.mu.gather(1, latent_index)
        return torch.where(
            active_valid[..., None], raw_mu, torch.zeros_like(raw_mu)
        )

    @torch.no_grad()
    def detokenize(
        self,
        raw_mu: torch.Tensor,
        local_root_motion: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        return self.decode(raw_mu, local_root_motion, local_root_valid_mask, frame_valid_mask)

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

        if not bool(torch.isfinite(latent_token).all()):
            raise ValueError("latent_token contains non-finite values")
        if not bool(torch.isfinite(local_root_patch).all()):
            raise ValueError("local_root_patch contains non-finite values")

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
        raw_token: torch.Tensor,
        local_root_patch: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        state: VAEDecoderState,
    ) -> tuple[VAEDecoderState, BodyPrediction]:
        """Decode one raw tokenizer token with explicit causal caches."""

        return self.decode_step(
            raw_token,
            local_root_patch,
            local_root_valid_mask,
            state,
        )


__all__ = ["BodyVAE"]
