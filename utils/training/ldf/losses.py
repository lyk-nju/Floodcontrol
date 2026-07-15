"""Loss reductions for LDF flow training."""

from __future__ import annotations

import torch

from utils.conditions.ldf import LDFPrediction
from utils.training.ldf.batch import LDFTrainingStep


def compute_velocity_loss(
    prediction: LDFPrediction,
    training_step: LDFTrainingStep,
    *,
    root_weight: float = 1.0,
    body_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute root/body flow-v MSE only on the active band."""

    mask = training_step.loss_mask
    if mask.dtype != torch.bool or not bool(mask.any()):
        raise ValueError("loss_mask must select at least one active token")
    root_error = (
        prediction.velocity.root_motion
        - training_step.target_velocity.root_motion
    ).square()
    body_error = (
        prediction.velocity.latent_motion
        - training_step.target_velocity.latent_motion
    ).square()
    root_loss = root_error[mask].mean()
    body_loss = body_error[mask].mean()
    total = float(root_weight) * root_loss + float(body_weight) * body_loss
    return {
        "anchor_root_flow_v": root_loss,
        "latent_body_flow_v": body_loss,
        "total": total,
    }


__all__ = ["compute_velocity_loss"]
