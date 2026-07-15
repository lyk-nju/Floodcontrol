"""Body VAE contracts.

The VAE operates on physical body features and owns normalization internally.
One latent token always represents exactly four consecutive motion frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from utils.motion_process import (
    BODY_CONTACT_DIM,
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    BODY_VELOCITY_DIM,
    LOCAL_ROOT_DIM,
    NUM_JOINTS,
    ROOT_DIM,
)
from utils.token_frame import FRAMES_PER_TOKEN, require_aligned_frame_count


def _require(name: str, value: torch.Tensor, shape_tail: tuple[int, ...]) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(f"{name} must end in {shape_tail}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class VAEInput:
    body_motion: torch.Tensor
    root_motion: torch.Tensor
    frame_valid_mask: torch.Tensor
    previous_root_frame: torch.Tensor | None = None
    previous_root_valid_mask: torch.Tensor | None = None
    body_feature_valid_mask: torch.Tensor | None = None

    def validate(self) -> None:
        _require("body_motion", self.body_motion, (BODY_DIM,))
        _require("root_motion", self.root_motion, (ROOT_DIM,))
        if self.body_motion.ndim != 3 or self.root_motion.ndim != 3:
            raise ValueError("body_motion and root_motion must be [B,F,D]")
        if self.body_motion.shape[:2] != self.root_motion.shape[:2]:
            raise ValueError("body_motion and root_motion must share [B,F]")
        batch, frames = self.body_motion.shape[:2]
        require_aligned_frame_count(frames)
        if tuple(self.frame_valid_mask.shape) != (batch, frames):
            raise ValueError("frame_valid_mask must be [B,F]")
        if self.frame_valid_mask.dtype != torch.bool:
            raise TypeError("frame_valid_mask must be bool")
        if self.previous_root_frame is not None and tuple(self.previous_root_frame.shape) != (batch, ROOT_DIM):
            raise ValueError("previous_root_frame must be [B,5]")
        if self.previous_root_valid_mask is not None:
            if self.previous_root_frame is None:
                raise ValueError("previous_root_valid_mask requires previous_root_frame")
            if tuple(self.previous_root_valid_mask.shape) != (batch,) or self.previous_root_valid_mask.dtype != torch.bool:
                raise ValueError("previous_root_valid_mask must be bool [B]")
        if self.body_feature_valid_mask is not None:
            if tuple(self.body_feature_valid_mask.shape) != tuple(self.body_motion.shape):
                raise ValueError("body_feature_valid_mask must match body_motion")
            if self.body_feature_valid_mask.dtype != torch.bool:
                raise TypeError("body_feature_valid_mask must be bool")


@dataclass(frozen=True)
class VAEPosterior:
    mu: torch.Tensor
    logvar: torch.Tensor

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        noise = torch.randn(std.shape, device=std.device, dtype=std.dtype, generator=generator)
        return self.mu + std * noise


@dataclass(frozen=True)
class BodyPrediction:
    continuous_body: torch.Tensor
    contact_logits: torch.Tensor

    def validate(self) -> None:
        _require("continuous_body", self.continuous_body, (BODY_CONTINUOUS_DIM,))
        _require("contact_logits", self.contact_logits, (BODY_CONTACT_DIM,))
        if self.continuous_body.shape[:2] != self.contact_logits.shape[:2]:
            raise ValueError("continuous_body and contact_logits must share [B,F]")

    def body_motion(self, *, threshold: float | None = None) -> torch.Tensor:
        contacts = self.contact_logits.sigmoid()
        if threshold is not None:
            contacts = (contacts >= float(threshold)).to(self.continuous_body.dtype)
        return torch.cat([self.continuous_body, contacts], dim=-1)


@dataclass(frozen=True)
class VAEPrediction:
    body: BodyPrediction
    posterior: VAEPosterior
    latent_sample: torch.Tensor
    local_root_motion: torch.Tensor
    local_root_valid_mask: torch.Tensor


@dataclass(frozen=True)
class VAEDecoderState:
    """Explicit causal decoder caches; no cache lives on the module."""

    caches: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def clone(self) -> "VAEDecoderState":
        return VAEDecoderState(
            tuple((first.clone(), second.clone()) for first, second in self.caches)
        )


__all__ = [
    "BODY_CONTACT_DIM", "BODY_CONTINUOUS_DIM", "BODY_DIM", "BODY_POSITION_DIM",
    "BODY_ROTATION_DIM", "BODY_VELOCITY_DIM",
    "LOCAL_ROOT_DIM", "NUM_JOINTS", "ROOT_DIM",
    "BodyPrediction", "VAEDecoderState", "VAEInput", "VAEPosterior", "VAEPrediction",
]
