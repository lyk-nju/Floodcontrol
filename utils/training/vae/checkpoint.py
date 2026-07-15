"""Shared BodyVAE checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch


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
    # Older checkpoints stored placeholder latent normalization as persistent
    # buffers. Latent statistics now come only from latent_stats_path.
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
    model,
    checkpoint_path: str | Path,
    *,
    use_ema: bool = True,
    freeze: bool = True,
):
    """Load one training checkpoint into ``model``.

    Physical statistics must match the model configuration. Latent statistics
    are deliberately absent from checkpoint state and remain untouched.
    """

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False, mmap=True
    )
    state = _checkpoint_state(checkpoint)
    configured = model.state_dict()
    for name in PHYSICAL_STATISTIC_BUFFERS:
        if name not in state or not torch.equal(state[name], configured[name].cpu()):
            raise RuntimeError(
                f"VAE checkpoint statistics do not match model buffer {name!r}"
            )
    if use_ema:
        state = _apply_ema(model, state, checkpoint)
    model.load_state_dict(state, strict=True)
    if freeze:
        model.eval().requires_grad_(False)
    return model


__all__ = ["PHYSICAL_STATISTIC_BUFFERS", "load_vae_checkpoint"]
