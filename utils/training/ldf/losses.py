"""Loss reductions for LDF flow training."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.conditions.ldf import LDFPrediction
from utils.training.ldf.steps import LDFTrainingStep
from utils.training.ldf.flow import endpoint_estimate


def _root_heading_losses(
    prediction: LDFPrediction,
    training_step: LDFTrainingStep,
    *,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
    beta_min: float,
    cosine_min_norm: float,
) -> dict[str, torch.Tensor]:
    """Supervise the pre-projection physical heading with beta compensation.

    The cosine term supplies circular geometry but has zero directional
    gradient at an exact antipode.  The raw-vector term retains a non-zero
    antipodal gradient and also keeps the pre-projection heading norm healthy.
    Both terms are weighted by ``1 / max(beta, beta_min)`` so that the
    endpoint-to-velocity chain factor does not suppress the commit boundary.
    """

    if float(beta_min) <= 0.0:
        raise ValueError("root_heading_beta_min must be positive")
    if float(cosine_min_norm) <= 0.0:
        raise ValueError("root_heading_cosine_min_norm must be positive")
    noisy_root = training_step.inputs.noisy_motion.root_motion
    velocity = prediction.velocity.root_motion
    target = training_step.clean_motion.root_motion
    if tuple(noisy_root.shape) != tuple(velocity.shape) or tuple(
        noisy_root.shape
    ) != tuple(target.shape):
        raise ValueError("noisy, predicted and target root must share shape")

    beta = training_step.inputs.beta.to(noisy_root)
    raw_clean_root = noisy_root + beta[..., None, None] * velocity
    mean = root_mean.to(device=noisy_root.device, dtype=torch.float32)
    std = root_std.to(device=noisy_root.device, dtype=torch.float32)
    raw_physical = raw_clean_root.float() * std + mean
    target_physical = target.float() * std + mean
    raw_heading = raw_physical[..., 3:5]
    target_heading = F.normalize(
        target_physical[..., 3:5],
        dim=-1,
        eps=1e-6,
    )
    raw_norm = raw_heading.norm(dim=-1, keepdim=True)
    safe_norm = raw_norm.clamp_min(float(cosine_min_norm))
    projected_heading = raw_heading / safe_norm
    cosine = (projected_heading * target_heading).sum(dim=-1).clamp(-1.0, 1.0)
    cosine_error = 1.0 - cosine
    vector_error = F.smooth_l1_loss(
        raw_heading,
        target_heading,
        reduction="none",
    ).mean(dim=-1)

    frame_mask = training_step.loss_mask[..., None].expand_as(cosine_error)
    cosine_valid = raw_norm.squeeze(-1).detach() >= float(cosine_min_norm)
    cosine_error = cosine_error * cosine_valid.to(cosine_error)
    beta_weight = beta.float().clamp_min(float(beta_min)).reciprocal()[..., None]
    beta_weight = beta_weight.expand_as(cosine_error)
    raw_norm = raw_norm.squeeze(-1)
    selected_norm = raw_norm[frame_mask]
    selected_cosine = cosine[frame_mask]
    selected_cosine_valid = cosine_valid[frame_mask]
    return {
        "cosine": cosine_error[frame_mask].mean(),
        "cosine_weighted": (cosine_error * beta_weight)[frame_mask].mean(),
        "vector": vector_error[frame_mask].mean(),
        "vector_weighted": (vector_error * beta_weight)[frame_mask].mean(),
        "raw_norm_mean": selected_norm.detach().mean(),
        "raw_norm_p10": torch.quantile(selected_norm.detach(), 0.1),
        "low_norm_ratio": (~selected_cosine_valid).float().mean(),
        "antipodal_ratio": (
            (selected_cosine.detach() < -0.9) & selected_cosine_valid
        ).float().mean(),
    }


def compute_velocity_loss(
    prediction: LDFPrediction,
    training_step: LDFTrainingStep,
    *,
    root_mean: torch.Tensor,
    root_std: torch.Tensor,
    root_weight: float = 1.0,
    body_weight: float = 1.0,
    root_heading_cosine_weight: float = 0.0,
    root_heading_vector_weight: float = 0.0,
    root_heading_beta_min: float = 0.1,
    root_heading_cosine_min_norm: float = 0.05,
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
    heading = _root_heading_losses(
        prediction,
        training_step,
        root_mean=root_mean,
        root_std=root_std,
        beta_min=float(root_heading_beta_min),
        cosine_min_norm=float(root_heading_cosine_min_norm),
    )
    total = (
        float(root_weight) * root_loss
        + float(body_weight) * body_loss
        + float(root_heading_cosine_weight) * heading["cosine_weighted"]
        + float(root_heading_vector_weight) * heading["vector_weighted"]
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
        "root_heading_cosine": heading["cosine"],
        "root_heading_cosine_weighted": heading["cosine_weighted"],
        "root_heading_vector": heading["vector"],
        "root_heading_vector_weighted": heading["vector_weighted"],
        "root_heading_raw_norm_mean": heading["raw_norm_mean"],
        "root_heading_raw_norm_p10": heading["raw_norm_p10"],
        "root_heading_low_norm_ratio": heading["low_norm_ratio"],
        "root_heading_antipodal_ratio": heading["antipodal_ratio"],
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
    root_heading_cosine_weight: float = 0.0,
    root_heading_vector_weight: float = 0.0,
    root_heading_beta_min: float = 0.1,
    root_heading_cosine_min_norm: float = 0.05,
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
    heading = _root_heading_losses(
        prediction,
        training_step,
        root_mean=root_mean,
        root_std=root_std,
        beta_min=float(root_heading_beta_min),
        cosine_min_norm=float(root_heading_cosine_min_norm),
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
        + float(root_heading_cosine_weight) * heading["cosine_weighted"]
        + float(root_heading_vector_weight) * heading["vector_weighted"]
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
        "root_heading_cosine": heading["cosine"],
        "root_heading_cosine_weighted": heading["cosine_weighted"],
        "root_heading_vector": heading["vector"],
        "root_heading_vector_weighted": heading["vector_weighted"],
        "root_heading_raw_norm_mean": heading["raw_norm_mean"],
        "root_heading_raw_norm_p10": heading["raw_norm_p10"],
        "root_heading_low_norm_ratio": heading["low_norm_ratio"],
        "root_heading_antipodal_ratio": heading["antipodal_ratio"],
        "root_boundary_displacement": boundary_loss,
        "total": total,
    }


__all__ = ["compute_offpath_loss", "compute_velocity_loss"]
