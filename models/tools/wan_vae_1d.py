"""Token-axis causal convolutional backbone for the body VAE.

Unlike the legacy frame-axis VAE, patch boundaries are explicit: four body
frames are flattened before the encoder and every decoder token projects to
exactly four output frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conditions.vae import (
    BODY_CONTACT_DIM,
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    FRAMES_PER_TOKEN,
    LOCAL_ROOT_DIM,
    BodyPrediction,
    VAEDecoderState,
    VAEPosterior,
)


class ChannelLayerNorm(nn.Module):
    """Layer normalization over channels without mixing token positions."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.transpose(1, 2)).transpose(1, 2)


class CausalConv1d(nn.Conv1d):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__(channels, channels, kernel_size=kernel_size, padding=0)
        self.cache_length = int(kernel_size) - 1

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(value, (self.cache_length, 0)))

    def init_cache(self, batch_size: int, *, device, dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.in_channels, self.cache_length, device=device, dtype=dtype)

    def stream_step(
        self, value: torch.Tensor, cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.shape[-1] != 1:
            raise ValueError("causal stream step accepts exactly one token")
        expected = (value.shape[0], self.in_channels, self.cache_length)
        if tuple(cache.shape) != expected:
            raise ValueError(f"invalid causal cache shape {tuple(cache.shape)}, expected {expected}")
        joined = torch.cat([cache.to(value), value], dim=-1)
        output = super().forward(joined)
        return output, joined[..., -self.cache_length :].clone()


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3, dropout: float = 0.0):
        super().__init__()
        self.norm1 = ChannelLayerNorm(channels)
        self.norm2 = ChannelLayerNorm(channels)
        self.conv1 = CausalConv1d(channels, kernel_size)
        self.conv2 = CausalConv1d(channels, kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return value + hidden

    def init_cache(self, batch_size: int, *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.conv1.init_cache(batch_size, device=device, dtype=dtype),
            self.conv2.init_cache(batch_size, device=device, dtype=dtype),
        )

    def stream_step(
        self,
        value: torch.Tensor,
        caches: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hidden, first = self.conv1.stream_step(F.silu(self.norm1(value)), caches[0])
        hidden, second = self.conv2.stream_step(
            self.dropout(F.silu(self.norm2(hidden))), caches[1]
        )
        return value + hidden, (first, second)


class CausalBodyVAE(nn.Module):
    """Body-only VAE with explicit four-frame patches and local-root decoder condition."""

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_input = nn.Linear(FRAMES_PER_TOKEN * BODY_DIM, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            [CausalResidualBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout)
             for _ in range(encoder_layers)]
        )
        self.posterior_head = nn.Linear(hidden_dim, latent_dim * 2)

        decoder_input_dim = latent_dim + FRAMES_PER_TOKEN * LOCAL_ROOT_DIM * 2
        self.decoder_input = nn.Linear(decoder_input_dim, hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            [CausalResidualBlock(hidden_dim, kernel_size=kernel_size, dropout=dropout)
             for _ in range(decoder_layers)]
        )
        self.continuous_head = nn.Linear(
            hidden_dim, FRAMES_PER_TOKEN * BODY_CONTINUOUS_DIM
        )
        self.contact_head = nn.Linear(hidden_dim, FRAMES_PER_TOKEN * BODY_CONTACT_DIM)

    @staticmethod
    def _patch_body(body_motion: torch.Tensor) -> torch.Tensor:
        if body_motion.ndim != 3 or body_motion.shape[-1] != BODY_DIM:
            raise ValueError("body_motion must be [B,F,265]")
        if body_motion.shape[1] % FRAMES_PER_TOKEN:
            raise ValueError("frame length must be divisible by four")
        return body_motion.reshape(body_motion.shape[0], -1, FRAMES_PER_TOKEN * BODY_DIM)

    def encode(self, normalized_body: torch.Tensor) -> VAEPosterior:
        hidden = self.encoder_input(self._patch_body(normalized_body)).transpose(1, 2)
        for block in self.encoder_blocks:
            hidden = block(hidden)
        mu, logvar = self.posterior_head(hidden.transpose(1, 2)).chunk(2, dim=-1)
        return VAEPosterior(mu=mu, logvar=logvar.clamp(min=-20.0, max=10.0))

    @staticmethod
    def _decoder_features(
        latent: torch.Tensor,
        normalized_local_root: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError("latent must be [B,T,D]")
        expected = (*latent.shape[:2], FRAMES_PER_TOKEN, LOCAL_ROOT_DIM)
        if tuple(normalized_local_root.shape) != expected:
            raise ValueError(f"local root must have shape {expected}")
        if tuple(local_root_valid_mask.shape) != expected or local_root_valid_mask.dtype != torch.bool:
            raise ValueError("local_root_valid_mask must be bool and match local root")
        local = torch.where(
            local_root_valid_mask, normalized_local_root, torch.zeros_like(normalized_local_root)
        )
        return torch.cat(
            [latent, local.flatten(2), local_root_valid_mask.flatten(2).to(latent.dtype)], dim=-1
        )

    def _project_output(self, hidden: torch.Tensor) -> BodyPrediction:
        hidden = hidden.transpose(1, 2)
        batch, tokens = hidden.shape[:2]
        continuous = self.continuous_head(hidden).reshape(
            batch, tokens * FRAMES_PER_TOKEN, BODY_CONTINUOUS_DIM
        )
        contacts = self.contact_head(hidden).reshape(
            batch, tokens * FRAMES_PER_TOKEN, BODY_CONTACT_DIM
        )
        return BodyPrediction(continuous, contacts)

    def decode(
        self,
        latent: torch.Tensor,
        normalized_local_root: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
    ) -> BodyPrediction:
        hidden = self.decoder_input(
            self._decoder_features(latent, normalized_local_root, local_root_valid_mask)
        ).transpose(1, 2)
        for block in self.decoder_blocks:
            hidden = block(hidden)
        return self._project_output(hidden)

    def init_decoder_state(self, batch_size: int, *, device, dtype) -> VAEDecoderState:
        return VAEDecoderState(
            tuple(block.init_cache(batch_size, device=device, dtype=dtype)
                  for block in self.decoder_blocks),
            token_index=0,
        )

    def stream_decode_step(
        self,
        latent_token: torch.Tensor,
        normalized_local_root_patch: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        state: VAEDecoderState,
    ) -> tuple[VAEDecoderState, BodyPrediction]:
        if latent_token.ndim != 3 or latent_token.shape[1] != 1:
            raise ValueError("latent_token must be [B,1,D]")
        if len(state.caches) != len(self.decoder_blocks):
            raise ValueError("decoder state does not match decoder depth")
        hidden = self.decoder_input(
            self._decoder_features(
                latent_token, normalized_local_root_patch, local_root_valid_mask
            )
        ).transpose(1, 2)
        caches = []
        for block, block_cache in zip(self.decoder_blocks, state.caches, strict=True):
            hidden, next_cache = block.stream_step(hidden, block_cache)
            caches.append(next_cache)
        prediction = self._project_output(hidden)
        return VAEDecoderState(tuple(caches), state.token_index + 1), prediction


__all__ = [
    "CausalBodyVAE", "CausalConv1d", "CausalResidualBlock"
]
