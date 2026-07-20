"""Loss reductions for LDF flow training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.conditions.ldf import LDFPrediction
from utils.training.ldf.steps import LDFTrainingStep
from utils.training.ldf.flow import endpoint_estimate


def _root_heading_cosine_loss(
    prediction: LDFPrediction,
    training_step: LDFTrainingStep,
    *,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
) -> torch.Tensor:
    """Compare the projected clean Root heading directly with GT heading."""

    predicted = prediction.clean_root_motion
    target = training_step.clean_motion.root_motion
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError("predicted and target clean root must share shape")
    mean = root_mean.to(predicted)
    std = root_std.to(predicted)
    predicted_heading = F.normalize(
        (predicted * std + mean)[..., 3:5].float(),
        dim=-1,
        eps=1e-6,
    )
    target_heading = F.normalize(
        (target * std + mean)[..., 3:5].float(),
        dim=-1,
        eps=1e-6,
    )
    cosine = (predicted_heading * target_heading).sum(dim=-1).clamp(-1.0, 1.0)
    error = 1.0 - cosine
    frame_mask = training_step.loss_mask[..., None].expand_as(error)
    return error[frame_mask].mean()


def compute_velocity_loss(
    prediction: LDFPrediction,
    training_step: LDFTrainingStep,
    *,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
    root_weight: float = 1.0,
    body_weight: float = 1.0,
    root_heading_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Compute root/body flow-v MSE only on the active band."""

    mask = training_step.loss_mask
    if mask.dtype != torch.bool:
        raise TypeError("loss_mask must be bool")
    root_error = (
        prediction.velocity.root_motion
        - training_step.target_velocity.root_motion
    ).square()
    body_error = (
        prediction.velocity.latent_motion
        - training_step.target_velocity.latent_motion
    ).square()
    root_loss = root_error[mask].mean()
    root_xz_loss = root_error[..., [0, 2]][mask].mean()
    root_height_loss = root_error[..., 1:2][mask].mean()
    root_heading_loss = root_error[..., 3:5][mask].mean()
    body_loss = body_error[mask].mean()
    heading_cosine = _root_heading_cosine_loss(
        prediction,
        training_step,
        root_mean=root_mean,
        root_std=root_std,
    )
    total = (
        float(root_weight) * root_loss
        + float(body_weight) * body_loss
        + float(root_heading_weight) * heading_cosine
    )
    zero = total.detach() * 0.0
    return {
        "anchor_root_flow_v": root_loss,
        "anchor_root_flow_xz": root_xz_loss,
        "anchor_root_flow_height": root_height_loss,
        "anchor_root_flow_heading": root_heading_loss,
        "latent_body_flow_v": body_loss,
        "anchor_root_offpath_endpoint": zero,
        "anchor_root_offpath_xz": zero,
        "anchor_root_offpath_height": zero,
        "anchor_root_offpath_heading": zero,
        "latent_body_offpath_endpoint": zero,
        "root_heading_cosine": heading_cosine,
        "root_boundary_displacement": zero,
        "total": total,
    }


def _masked_endpoint_loss(
    error: torch.Tensor,
    mask: torch.Tensor,
    beta: torch.Tensor,
    beta_min: float,
) -> torch.Tensor:
    weights = beta.clamp_min(float(beta_min)).square().reciprocal()
    while weights.ndim < error.ndim:
        weights = weights.unsqueeze(-1)
    weighted = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none")
    weighted = weighted * weights.to(weighted)
    return weighted[mask].mean()


def _root_boundary_displacement_loss(
    endpoint_root: torch.Tensor,
    target_root: torch.Tensor,
    training_step: LDFTrainingStep,
    *,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
) -> torch.Tensor:
    mean = root_mean.to(endpoint_root)
    std = root_std.to(endpoint_root)
    predicted = endpoint_root * std + mean
    target = target_root * std + mean
    predicted = predicted.flatten(1, 2)
    target = target.flatten(1, 2)

    valid_tokens = training_step.inputs.history_mask | training_step.loss_mask
    active_frames = training_step.loss_mask.repeat_interleave(
        endpoint_root.shape[2], dim=1
    )
    valid_frames = valid_tokens.repeat_interleave(endpoint_root.shape[2], dim=1)
    transition_mask = active_frames[:, 1:] & valid_frames[:, :-1]
    if not bool(transition_mask.any()):
        return endpoint_root.sum() * 0.0
    predicted_delta = predicted[:, 1:, [0, 2]] - predicted[:, :-1, [0, 2]]
    target_delta = target[:, 1:, [0, 2]] - target[:, :-1, [0, 2]]
    return F.smooth_l1_loss(
        predicted_delta[transition_mask],
        target_delta[transition_mask],
    )


def compute_offpath_loss(
    prediction: LDFPrediction,
    training_step: LDFTrainingStep,
    *,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
    root_weight: float = 1.0,
    body_weight: float = 1.0,
    rollout_weight: float = 1.0,
    offpath_beta_min: float = 0.1,
    root_boundary_weight: float = 0.0,
    root_heading_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Stabilize the clean endpoint from a persistent off-path solver state."""

    if not training_step.is_rollout:
        raise ValueError("off-path loss requires a persistent rollout step")
    if float(offpath_beta_min) <= 0.0:
        raise ValueError("offpath_beta_min must be positive")
    endpoint = endpoint_estimate(
        training_step.inputs.noisy_motion,
        training_step.inputs.beta,
        prediction.velocity,
    )
    root_error = endpoint.root_motion - training_step.clean_motion.root_motion
    body_error = endpoint.latent_motion - training_step.clean_motion.latent_motion
    mask = training_step.loss_mask
    root_loss = _masked_endpoint_loss(
        root_error,
        mask,
        training_step.inputs.beta,
        float(offpath_beta_min),
    )
    root_xz_loss = _masked_endpoint_loss(
        root_error[..., [0, 2]],
        mask,
        training_step.inputs.beta,
        float(offpath_beta_min),
    )
    root_height_loss = _masked_endpoint_loss(
        root_error[..., 1:2],
        mask,
        training_step.inputs.beta,
        float(offpath_beta_min),
    )
    root_heading_loss = _masked_endpoint_loss(
        root_error[..., 3:5],
        mask,
        training_step.inputs.beta,
        float(offpath_beta_min),
    )
    body_loss = _masked_endpoint_loss(
        body_error,
        mask,
        training_step.inputs.beta,
        float(offpath_beta_min),
    )
    heading_cosine = _root_heading_cosine_loss(
        prediction,
        training_step,
        root_mean=root_mean,
        root_std=root_std,
    )
    if float(root_boundary_weight) != 0.0:
        boundary_loss = _root_boundary_displacement_loss(
            endpoint.root_motion,
            training_step.clean_motion.root_motion,
            training_step,
            root_mean=root_mean,
            root_std=root_std,
        )
    else:
        boundary_loss = root_loss.detach() * 0.0
    total = float(rollout_weight) * (
        float(root_weight) * root_loss
        + float(body_weight) * body_loss
        + float(root_heading_weight) * heading_cosine
    ) + float(root_boundary_weight) * boundary_loss
    zero = total.detach() * 0.0
    return {
        "anchor_root_flow_v": zero,
        "anchor_root_flow_xz": zero,
        "anchor_root_flow_height": zero,
        "anchor_root_flow_heading": zero,
        "latent_body_flow_v": zero,
        "anchor_root_offpath_endpoint": root_loss,
        "anchor_root_offpath_xz": root_xz_loss,
        "anchor_root_offpath_height": root_height_loss,
        "anchor_root_offpath_heading": root_heading_loss,
        "latent_body_offpath_endpoint": body_loss,
        "root_heading_cosine": heading_cosine,
        "root_boundary_displacement": boundary_loss,
        "total": total,
    }


__all__ = ["compute_offpath_loss", "compute_velocity_loss"]
