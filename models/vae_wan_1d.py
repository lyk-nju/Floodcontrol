"""Public body VAE wrapper."""

from __future__ import annotations

import json
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from models.tools.wan_vae_1d import CausalBodyVAE
from utils.conditions.vae import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    CONTRACT_VERSION,
    FRAMES_PER_TOKEN,
    LOCAL_ROOT_DIM,
    BodyPrediction,
    VAEDecoderState,
    VAEInput,
    VAEPrediction,
    VAEPosterior,
)
from utils.motion_representation import MotionStatistics, derive_patched_local_root


def _stat_tensor(name: str, value, dim: int, *, positive: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != (dim,):
        raise ValueError(f"{name} must have shape [{dim}], got {tuple(tensor.shape)}")
    if positive and bool((tensor <= 0).any()):
        raise ValueError(f"{name} must be strictly positive")
    return tensor


class BodyVAE(nn.Module):
    """Body-only VAE; physical values cross the public boundary."""

    contract_version = CONTRACT_VERSION

    def __init__(
        self,
        *,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.0,
        fps: float = 20.0,
        motion_stats_path: str | None = None,
        latent_stats_path: str | None = None,
        body_cont_mean=None,
        body_cont_std=None,
        local_root_mean=None,
        local_root_std=None,
        latent_mean=None,
        latent_std=None,
        allow_identity_statistics: bool = False,
        require_latent_statistics: bool = True,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.fps = float(fps)
        if motion_stats_path is not None:
            stats = MotionStatistics.load(motion_stats_path)
            body_cont_mean = stats.body_cont_mean
            body_cont_std = stats.body_cont_std
            local_root_mean = stats.local_root_mean
            local_root_std = stats.local_root_std
        if latent_stats_path is not None:
            with np.load(latent_stats_path, allow_pickle=False) as data:
                latent_mean = data["latent_mu_mean"]
                latent_std = data["latent_mu_std"]
                metadata = json.loads(str(data["metadata"]))
            if metadata.get("contract_version") != CONTRACT_VERSION:
                raise ValueError("latent statistics contract version mismatch")
            if int(metadata.get("latent_dim", -1)) != self.latent_dim:
                raise ValueError("latent statistics dimension mismatch")
        physical_missing = [name for name, value in (
            ("body_cont_mean", body_cont_mean), ("body_cont_std", body_cont_std),
            ("local_root_mean", local_root_mean), ("local_root_std", local_root_std),
        ) if value is None]
        latent_missing = latent_mean is None or latent_std is None
        missing = physical_missing + (
            ["latent_mean", "latent_std"] if latent_missing and require_latent_statistics else []
        )
        if missing and not allow_identity_statistics:
            raise ValueError(
                "BodyVAE requires explicit real statistics; missing " + ", ".join(missing)
            )
        if physical_missing and not allow_identity_statistics:
            raise ValueError("physical VAE statistics may not use identity fallback")
        if allow_identity_statistics:
            body_cont_mean = torch.zeros(BODY_CONTINUOUS_DIM) if body_cont_mean is None else body_cont_mean
            body_cont_std = torch.ones(BODY_CONTINUOUS_DIM) if body_cont_std is None else body_cont_std
            local_root_mean = torch.zeros(LOCAL_ROOT_DIM) if local_root_mean is None else local_root_mean
            local_root_std = torch.ones(LOCAL_ROOT_DIM) if local_root_std is None else local_root_std
            latent_mean = torch.zeros(self.latent_dim) if latent_mean is None else latent_mean
            latent_std = torch.ones(self.latent_dim) if latent_std is None else latent_std
        elif latent_missing and not require_latent_statistics:
            latent_mean = torch.zeros(self.latent_dim)
            latent_std = torch.ones(self.latent_dim)
        self.latent_statistics_ready = bool(allow_identity_statistics or not latent_missing)
        self.register_buffer("body_cont_mean", _stat_tensor("body_cont_mean", body_cont_mean, BODY_CONTINUOUS_DIM))
        self.register_buffer("body_cont_std", _stat_tensor("body_cont_std", body_cont_std, BODY_CONTINUOUS_DIM, positive=True))
        self.register_buffer("local_root_mean", _stat_tensor("local_root_mean", local_root_mean, LOCAL_ROOT_DIM))
        self.register_buffer("local_root_std", _stat_tensor("local_root_std", local_root_std, LOCAL_ROOT_DIM, positive=True))
        self.register_buffer("latent_mean", _stat_tensor("latent_mean", latent_mean, self.latent_dim))
        self.register_buffer("latent_std", _stat_tensor("latent_std", latent_std, self.latent_dim, positive=True))
        self.model = CausalBodyVAE(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def normalize_body(self, body_motion: torch.Tensor) -> torch.Tensor:
        if body_motion.shape[-1] != BODY_DIM:
            raise ValueError("body_motion must end in 265")
        continuous = (body_motion[..., :BODY_CONTINUOUS_DIM] - self.body_cont_mean) / self.body_cont_std
        contacts = body_motion[..., BODY_CONTINUOUS_DIM:]
        return torch.cat([continuous, contacts], dim=-1)

    def normalize_local_root(self, local_root_motion: torch.Tensor) -> torch.Tensor:
        return (local_root_motion - self.local_root_mean) / self.local_root_std

    def normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.latent_statistics_ready:
            raise RuntimeError("latent mu statistics must be computed after freezing the VAE")
        return (latent - self.latent_mean) / self.latent_std

    def unnormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.latent_statistics_ready:
            raise RuntimeError("latent mu statistics must be loaded before LDF detokenization")
        return latent * self.latent_std + self.latent_mean

    def encode(
        self, body_motion: torch.Tensor, frame_valid_mask: torch.Tensor
    ) -> VAEPosterior:
        if body_motion.ndim != 3 or body_motion.shape[-1] != BODY_DIM:
            raise ValueError("body_motion must be [B,F,265]")
        if tuple(frame_valid_mask.shape) != tuple(body_motion.shape[:2]) or frame_valid_mask.dtype != torch.bool:
            raise ValueError("frame_valid_mask must be bool [B,F]")
        normalized = self.normalize_body(body_motion)
        normalized = torch.where(frame_valid_mask[..., None], normalized, torch.zeros_like(normalized))
        return self.model.encode(normalized)

    def decode(
        self,
        latent_tokens: torch.Tensor,
        local_root_motion: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        normalized_root = self.normalize_local_root(local_root_motion)
        output = self.model.decode(latent_tokens, normalized_root, local_root_valid_mask)
        physical = output.continuous_body * self.body_cont_std + self.body_cont_mean
        if frame_valid_mask is not None:
            if tuple(frame_valid_mask.shape) != tuple(physical.shape[:2]):
                raise ValueError("frame_valid_mask does not match decoded frames")
            physical = torch.where(frame_valid_mask[..., None], physical, torch.zeros_like(physical))
        result = BodyPrediction(physical, output.contact_logits)
        result.validate()
        return result

    def forward(self, inputs: VAEInput | Mapping[str, torch.Tensor]) -> VAEPrediction:
        if not isinstance(inputs, VAEInput):
            inputs = VAEInput(
                body_motion=inputs["body_motion"],
                root_motion=inputs["root_motion"],
                frame_valid_mask=inputs["frame_valid_mask"],
                previous_root_frame=inputs.get("previous_root_frame"),
                previous_root_valid_mask=inputs.get("previous_root_valid_mask"),
                body_feature_valid_mask=inputs.get("body_feature_valid_mask"),
            )
        inputs.validate()
        posterior = self.encode(inputs.body_motion, inputs.frame_valid_mask)
        latent = posterior.sample()
        local_root, local_valid = derive_patched_local_root(
            inputs.root_motion, inputs.previous_root_frame, fps=self.fps,
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
    def tokenize(self, body_motion: torch.Tensor, frame_valid_mask: torch.Tensor) -> torch.Tensor:
        return self.normalize_latent(self.encode(body_motion, frame_valid_mask).mu)

    @torch.no_grad()
    def detokenize(
        self,
        normalized_mu: torch.Tensor,
        local_root_motion: torch.Tensor,
        local_root_valid_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> BodyPrediction:
        return self.decode(
            self.unnormalize_latent(normalized_mu), local_root_motion,
            local_root_valid_mask, frame_valid_mask
        )

    def init_decoder_state(
        self, batch_size: int, *, device=None, dtype=torch.float32
    ) -> VAEDecoderState:
        device = torch.device(device or self.body_cont_mean.device)
        return self.model.init_decoder_state(batch_size, device=device, dtype=dtype)

    @torch.no_grad()
    def stream_decode_step(
        self,
        latent_token: torch.Tensor,
        local_root_patch: torch.Tensor,
        state: VAEDecoderState,
        local_root_valid_mask: torch.Tensor | None = None,
        *,
        normalized_latent: bool = True,
    ) -> tuple[VAEDecoderState, BodyPrediction]:
        if local_root_valid_mask is None:
            local_root_valid_mask = torch.ones_like(local_root_patch, dtype=torch.bool)
        latent = self.unnormalize_latent(latent_token) if normalized_latent else latent_token
        next_state, output = self.model.stream_decode_step(
            latent,
            self.normalize_local_root(local_root_patch),
            local_root_valid_mask,
            state,
        )
        physical = output.continuous_body * self.body_cont_std + self.body_cont_mean
        return next_state, BodyPrediction(physical, output.contact_logits)

    @staticmethod
    def snapshot_decoder_state(state: VAEDecoderState) -> dict:
        return {
            "contract_version": CONTRACT_VERSION,
            "token_index": int(state.token_index),
            "caches": tuple((first.detach().clone(), second.detach().clone())
                            for first, second in state.caches),
        }

    @staticmethod
    def restore_decoder_state(snapshot: Mapping[str, object]) -> VAEDecoderState:
        if snapshot.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("decoder snapshot contract version mismatch")
        caches = snapshot.get("caches")
        if not isinstance(caches, (tuple, list)):
            raise TypeError("decoder snapshot caches are missing")
        return VAEDecoderState(
            tuple((first.clone(), second.clone()) for first, second in caches),
            int(snapshot["token_index"]),
        )


__all__ = ["BodyVAE"]
