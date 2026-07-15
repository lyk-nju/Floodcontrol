"""Shared immutable model ownership for the Web process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from models.diffusion_forcing_wan import LDF
from models.vae_wan_1d import BodyVAE
from utils.inference import GuidanceConfig, InferenceConfig, InferenceSession


TextEncoder = Callable[[list[str], torch.device], list[torch.Tensor]]


@dataclass(frozen=True)
class ModelBundle:
    """Process-level evaluation models shared by isolated inference sessions."""

    ldf: LDF
    body_vae: BodyVAE
    text_encoder: TextEncoder
    device: torch.device

    def __post_init__(self) -> None:
        if not isinstance(self.ldf, LDF) or not isinstance(self.body_vae, BodyVAE):
            raise TypeError("ModelBundle requires public LDF and BodyVAE instances")
        if self.ldf.training or self.body_vae.training:
            raise ValueError("Web models must be in evaluation mode")
        if not callable(self.text_encoder):
            raise TypeError("text_encoder must be callable")
        object.__setattr__(self, "device", torch.device(self.device))

    def create_session(
        self,
        *,
        config: InferenceConfig,
        guidance: GuidanceConfig,
        seed: int,
        initial_world_xz,
        initial_yaw: float | None,
        initial_text: str,
    ) -> InferenceSession:
        return InferenceSession(
            ldf=self.ldf,
            body_vae=self.body_vae,
            text_encoder=self.text_encoder,
            config=config,
            guidance=guidance,
            seed=seed,
            initial_world_xz=initial_world_xz,
            initial_yaw=initial_yaw,
            initial_text=initial_text,
        )


__all__ = ["ModelBundle", "TextEncoder"]
