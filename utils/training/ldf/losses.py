"""Losses for independent Root/Body LDF prediction contracts."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.conditions.ldf import LDFPrediction
from utils.training.ldf.steps import LDFTrainingStep


def _require_active_mask(step: LDFTrainingStep) -> tuple[torch.Tensor, torch.Tensor]:
    mask = step.loss_mask
    if mask.dtype != torch.bool:
        raise TypeError("loss_mask must be bool")
    if not bool(mask.any()):
        raise ValueError("loss_mask must select at least one active token")
    beta = step.inputs.beta
    if bool((beta[mask] <= 0).any()):
        raise ValueError("active loss positions must have strictly positive beta")
    return mask, beta


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value[mask].mean()


def _block_losses(
    error: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce root5 errors as equally weighted XZ/height/heading blocks."""

    xz = _masked_mean(error[..., [0, 2]], mask)
    height = _masked_mean(error[..., 1:2], mask)
    heading = _masked_mean(error[..., 3:5], mask)
    return xz + height + heading, xz, height, heading


def compute_root_x0_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
) -> dict[str, torch.Tensor]:
    """Supervise the unprojected physical root x0 in three equal blocks."""

    mask, _ = _require_active_mask(step)
    raw = prediction.raw_root_output
    target = step.clean_motion.root_motion
    error = F.smooth_l1_loss(raw, target, reduction="none")
    total, xz, height, heading = _block_losses(error, mask)
    return {
        "root": total,
        "root_xz": xz,
        "root_height": height,
        "root_heading": heading,
    }


def compute_root_ideal_velocity_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
) -> dict[str, torch.Tensor]:
    """Supervise ideal-bridge root velocity with x0 - fixed_noise."""

    mask, _ = _require_active_mask(step)
    target = step.clean_motion.root_motion - step.noise.root_motion
    error = (prediction.raw_root_output - target).square()
    total, xz, height, heading = _block_losses(error, mask)
    return {
        "root": total,
        "root_xz": xz,
        "root_height": height,
        "root_heading": heading,
    }


def _corrective_velocity_error(
    current: torch.Tensor,
    beta: torch.Tensor,
    raw_velocity: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return (current + beta*v - target) / beta on active positions.

    The division uses the real scheduler beta.  History is never divided, and
    an active beta of zero is a contract error rather than a value to clamp.
    """

    if bool((beta[mask] <= 0).any()):
        raise ValueError("corrective velocity requires strictly positive active beta")
    safe_beta = torch.where(mask, beta, torch.ones_like(beta))
    while safe_beta.ndim < current.ndim:
        safe_beta = safe_beta.unsqueeze(-1)
    return (current + safe_beta.to(current) * raw_velocity - target) / safe_beta.to(
        current
    )


def compute_root_corrective_velocity_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
) -> dict[str, torch.Tensor]:
    """Supervise root velocity on a persistent off-path solver state."""

    mask, beta = _require_active_mask(step)
    corrective_error = _corrective_velocity_error(
        step.inputs.noisy_motion.root_motion,
        beta,
        prediction.raw_root_output,
        step.clean_motion.root_motion,
        mask,
    )
    error = F.smooth_l1_loss(
        corrective_error, torch.zeros_like(corrective_error), reduction="none"
    )
    total, xz, height, heading = _block_losses(error, mask)
    return {
        "root": total,
        "root_xz": xz,
        "root_height": height,
        "root_heading": heading,
    }


def compute_body_x0_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
) -> torch.Tensor:
    mask, _ = _require_active_mask(step)
    error = F.smooth_l1_loss(
        prediction.raw_body_output,
        step.clean_motion.latent_motion,
        reduction="none",
    )
    return _masked_mean(error, mask)


def compute_body_ideal_velocity_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
) -> torch.Tensor:
    """Supervise ideal-bridge Body velocity with x0 - fixed_noise."""

    mask, _ = _require_active_mask(step)
    target = step.clean_motion.latent_motion - step.noise.latent_motion
    return _masked_mean((prediction.raw_body_output - target).square(), mask)


def compute_body_corrective_velocity_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
) -> torch.Tensor:
    """Supervise Body velocity needed by the current persistent state."""

    mask, beta = _require_active_mask(step)
    corrective_error = _corrective_velocity_error(
        step.inputs.noisy_motion.latent_motion,
        beta,
        prediction.raw_body_output,
        step.clean_motion.latent_motion,
        mask,
    )
    error = F.smooth_l1_loss(
        corrective_error, torch.zeros_like(corrective_error), reduction="none"
    )
    return _masked_mean(error, mask)


def _root_boundary_displacement_loss(
    predicted_root: torch.Tensor,
    target_root: torch.Tensor,
    step: LDFTrainingStep,
) -> torch.Tensor:
    """Compare physical XZ frame displacements around active boundaries."""

    predicted = predicted_root.flatten(1, 2)
    target = target_root.flatten(1, 2)
    valid_tokens = step.inputs.history_mask | step.loss_mask
    active_frames = step.loss_mask.repeat_interleave(predicted_root.shape[2], dim=1)
    valid_frames = valid_tokens.repeat_interleave(predicted_root.shape[2], dim=1)
    transition_mask = active_frames[:, 1:] & valid_frames[:, :-1]
    if not bool(transition_mask.any()):
        return predicted_root.sum() * 0.0
    predicted_delta = predicted[:, 1:, [0, 2]] - predicted[:, :-1, [0, 2]]
    target_delta = target[:, 1:, [0, 2]] - target[:, :-1, [0, 2]]
    return F.smooth_l1_loss(
        predicted_delta[transition_mask], target_delta[transition_mask]
    )


def _root_heading_metrics(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
    *,
    root_prediction_type: str,
    include_detailed_metrics: bool,
) -> dict[str, torch.Tensor]:
    """Observe raw clean heading angle/norm without contributing to loss."""

    mask, beta = _require_active_mask(step)
    if root_prediction_type == "x0":
        raw_clean = prediction.raw_root_output
    elif root_prediction_type == "velocity":
        raw_clean = (
            step.inputs.noisy_motion.root_motion
            + beta[..., None, None].to(prediction.raw_root_output)
            * prediction.raw_root_output
        )
    else:
        raise ValueError(f"unsupported root prediction type {root_prediction_type!r}")
    predicted_heading = raw_clean[..., 3:5].float()
    target_heading = step.clean_motion.root_motion[..., 3:5].float()
    predicted_norm = predicted_heading.norm(dim=-1)
    predicted_unit = F.normalize(predicted_heading, dim=-1, eps=1e-6)
    target_unit = F.normalize(target_heading, dim=-1, eps=1e-6)
    cosine = (predicted_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
    angle_degrees = torch.rad2deg(torch.acos(cosine))
    frame_mask = mask[..., None].expand_as(cosine)
    selected_norm = predicted_norm[frame_mask]
    result = {
        "heading_bias": angle_degrees[frame_mask].mean().detach(),
        "heading_norm": selected_norm.mean().detach(),
        "heading_antipodal_ratio": (
            cosine[frame_mask] < -0.9
        ).float().mean().detach(),
    }
    if include_detailed_metrics:
        result.update(
            {
                "heading_norm_p10": torch.quantile(
                    selected_norm, 0.1
                ).detach(),
                "heading_low_norm_ratio": (
                    selected_norm < 0.25
                ).float().mean().detach(),
            }
        )
    return result


def compute_ldf_loss(
    prediction: LDFPrediction,
    step: LDFTrainingStep,
    *,
    root_prediction_type: str,
    body_prediction_type: str,
    root_weight: float = 1.0,
    body_weight: float = 1.0,
    rollout_weight: float = 1.0,
    root_boundary_weight: float = 0.0,
    include_detailed_metrics: bool = True,
) -> dict[str, torch.Tensor]:
    """Dispatch Root/Body objectives and return the compact public log contract."""

    if root_prediction_type == "x0":
        root = compute_root_x0_loss(prediction, step)
    elif root_prediction_type == "velocity" and step.is_rollout:
        root = compute_root_corrective_velocity_loss(prediction, step)
    elif root_prediction_type == "velocity":
        root = compute_root_ideal_velocity_loss(prediction, step)
    else:
        raise ValueError(f"unsupported root prediction type {root_prediction_type!r}")

    if body_prediction_type == "x0":
        body = compute_body_x0_loss(prediction, step)
    elif body_prediction_type == "velocity" and step.is_rollout:
        body = compute_body_corrective_velocity_loss(prediction, step)
    elif body_prediction_type == "velocity":
        body = compute_body_ideal_velocity_loss(prediction, step)
    else:
        raise ValueError(f"unsupported body prediction type {body_prediction_type!r}")

    boundary = (
        _root_boundary_displacement_loss(
            prediction.clean_motion.root_motion,
            step.clean_motion.root_motion,
            step,
        )
        if float(root_boundary_weight) != 0.0
        else root["root"].detach() * 0.0
    )
    scale = float(rollout_weight) if step.is_rollout else 1.0
    weighted_root = scale * float(root_weight) * root["root"]
    weighted_body = scale * float(body_weight) * body
    total = (
        weighted_root
        + weighted_body
        + float(root_boundary_weight) * boundary
    )
    result = {
        "loss_root": weighted_root,
        "loss_body": weighted_body,
        "loss_root_xz": root["root_xz"],
        "loss_root_height": root["root_height"],
        "loss_root_heading": root["root_heading"],
        "total": total,
    }
    result.update(
        _root_heading_metrics(
            prediction,
            step,
            root_prediction_type=root_prediction_type,
            include_detailed_metrics=include_detailed_metrics,
        )
    )
    return result


__all__ = [
    "compute_body_corrective_velocity_loss",
    "compute_body_ideal_velocity_loss",
    "compute_body_x0_loss",
    "compute_ldf_loss",
    "compute_root_corrective_velocity_loss",
    "compute_root_ideal_velocity_loss",
    "compute_root_x0_loss",
]
