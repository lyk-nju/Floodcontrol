"""Shared BodyVAE checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from models.vae_wan_1d import BodyVAE


PHYSICAL_STATISTIC_BUFFERS = (
    "body_cont_mean",
    "body_cont_std",
    "local_root_mean",
    "local_root_std",
)


def _checkpoint_state(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("VAE checkpoint is missing state_dict")
    result = {
        str(name): value.detach().cpu().clone()
        for name, value in state.items()
        if torch.is_tensor(value)
    }
    # Historical VAE checkpoints may contain post-hoc latent normalization.
    # Raw posterior means are now the tokenizer space, so these are not state.
    result.pop("latent_mean", None)
    result.pop("latent_std", None)
    return result


def _apply_ema(
    model,
    state: dict[str, torch.Tensor],
    checkpoint: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    ema_state = checkpoint.get("ema_state")
    if not isinstance(ema_state, Mapping):
        raise ValueError("VAE checkpoint is missing ema_state")
    shadows = ema_state.get("shadow_params")
    if not isinstance(shadows, (tuple, list)):
        raise ValueError("VAE ema_state is missing shadow_params")
    parameter_names = [name for name, _ in model.named_parameters()]
    if len(parameter_names) != len(shadows):
        raise ValueError(
            "EMA parameter count does not match BodyVAE: "
            f"{len(shadows)} != {len(parameter_names)}"
        )
    for name, shadow in zip(parameter_names, shadows, strict=True):
        if name not in state:
            raise ValueError(f"EMA parameter {name!r} is absent from checkpoint")
        state[name] = shadow.detach().cpu().clone()
    return state


def load_vae_checkpoint(
    checkpoint_path: str | Path,
    *,
    model_params: Mapping[str, object],
    use_ema: bool = True,
    freeze: bool = True,
) -> BodyVAE:
    """Construct a BodyVAE from one self-contained training checkpoint.

    Architecture parameters come from configuration. Physical body/local-root
    statistics come only from checkpoint buffers, so LDF/runtime callers do not
    need the VAE training statistics file.
    """

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
    )
    state = _checkpoint_state(checkpoint)
    statistics: dict[str, torch.Tensor] = {}
    for name in PHYSICAL_STATISTIC_BUFFERS:
        value = state.get(name)
        if not torch.is_tensor(value):
            raise RuntimeError(f"VAE checkpoint is missing physical buffer {name!r}")
        statistics[name] = value
    params = dict(model_params)
    forbidden = {"motion_stats_path", "motion_statistics", "latent_stats_path"}
    duplicated = forbidden.intersection(params)
    if duplicated:
        raise ValueError(
            "checkpoint-loaded BodyVAE parameters must contain architecture only; "
            f"remove {sorted(duplicated)}"
        )
    model = BodyVAE(motion_statistics=statistics, **params)
    if use_ema:
        state = _apply_ema(model, state, checkpoint)
    model.load_state_dict(state, strict=True)
    if freeze:
        model.eval().requires_grad_(False)
    return model


__all__ = ["PHYSICAL_STATISTIC_BUFFERS", "load_vae_checkpoint"]
