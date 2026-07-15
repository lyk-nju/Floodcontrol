"""Optimization losses for the body VAE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.conditions.vae import (
    BODY_CONTINUOUS_DIM,
    BODY_POSITION_DIM,
    BODY_ROTATION_DIM,
    NUM_JOINTS,
    VAEInput,
    VAEPrediction,
)
from utils.motion_process import rotation_to_matrix
from utils.token_frame import frame_valid_to_token_valid


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class VAELoss(nn.Module):
    def __init__(
        self,
        *,
        body_cont_mean,
        body_cont_std,
        lambda_position: float = 1.0,
        lambda_rotation: float = 1.0,
        lambda_velocity: float = 1.0,
        lambda_contact: float = 1.0,
        lambda_skating: float = 0.01,
        beta_kl: float = 1e-5,
        kl_warmup_steps: int = 0,
        foot_joint_indices: tuple[int, int, int, int] = (7, 10, 8, 11),
        lambda_geodesic: float = 0.0,
        lambda_fk: float = 0.0,
        lambda_position_consistency: float = 0.0,
        lambda_velocity_consistency: float = 0.0,
        skeleton_parents=None,
        skeleton_offsets=None,
        fps: float = 20.0,
    ):
        super().__init__()
        self.register_buffer("body_cont_mean", torch.as_tensor(body_cont_mean).float())
        self.register_buffer("body_cont_std", torch.as_tensor(body_cont_std).float())
        if tuple(self.body_cont_mean.shape) != (BODY_CONTINUOUS_DIM,) or tuple(self.body_cont_std.shape) != (BODY_CONTINUOUS_DIM,):
            raise ValueError("body continuous statistics must have shape [261]")
        if bool((self.body_cont_std <= 0).any()):
            raise ValueError("body continuous std must be positive")
        self.lambda_position = float(lambda_position)
        self.lambda_rotation = float(lambda_rotation)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_contact = float(lambda_contact)
        self.lambda_skating = float(lambda_skating)
        self.beta_kl = float(beta_kl)
        self.kl_warmup_steps = int(kl_warmup_steps)
        self.foot_joint_indices = tuple(int(index) for index in foot_joint_indices)
        self.lambda_geodesic = float(lambda_geodesic)
        self.lambda_fk = float(lambda_fk)
        self.lambda_position_consistency = float(lambda_position_consistency)
        self.lambda_velocity_consistency = float(lambda_velocity_consistency)
        self.fps = float(fps)
        if skeleton_parents is None or skeleton_offsets is None:
            self.register_buffer("skeleton_parents", torch.empty(0, dtype=torch.long))
            self.register_buffer("skeleton_offsets", torch.empty(0, 3))
        else:
            parents = torch.as_tensor(skeleton_parents, dtype=torch.long)
            offsets = torch.as_tensor(skeleton_offsets, dtype=torch.float32)
            if tuple(parents.shape) != (NUM_JOINTS,) or tuple(offsets.shape) != (NUM_JOINTS, 3):
                raise ValueError("skeleton parents/offsets must be [22] and [22,3]")
            self.register_buffer("skeleton_parents", parents)
            self.register_buffer("skeleton_offsets", offsets)
        if (self.lambda_fk or self.lambda_position_consistency) and self.skeleton_parents.numel() == 0:
            raise ValueError("FK losses require versioned skeleton_parents and skeleton_offsets")

    @staticmethod
    def _global_positions(root_motion: torch.Tensor, non_root_positions: torch.Tensor) -> torch.Tensor:
        positions = non_root_positions.clone()
        positions[..., 0] += root_motion[..., None, 0]
        positions[..., 2] += root_motion[..., None, 2]
        return torch.cat([root_motion[..., None, :3], positions], dim=-2)

    def _forward_kinematics(
        self, root_motion: torch.Tensor, global_rotations: torch.Tensor
    ) -> torch.Tensor:
        positions = [root_motion[..., :3]]
        for joint in range(1, NUM_JOINTS):
            parent = int(self.skeleton_parents[joint])
            if parent < 0 or parent >= joint:
                raise ValueError("skeleton parents must be topologically ordered")
            offset = torch.einsum(
                "...ij,j->...i", global_rotations[..., parent, :, :], self.skeleton_offsets[joint]
            )
            positions.append(positions[parent] + offset)
        return torch.stack(positions, dim=-2)

    def forward(
        self,
        inputs: VAEInput,
        prediction: VAEPrediction,
        *,
        global_step: int = 0,
    ) -> dict[str, torch.Tensor]:
        inputs.validate()
        predicted = prediction.body.continuous_body
        target = inputs.body_motion[..., :BODY_CONTINUOUS_DIM]
        normalized_pred = (predicted - self.body_cont_mean) / self.body_cont_std
        normalized_target = (target - self.body_cont_mean) / self.body_cont_std
        frame_mask = inputs.frame_valid_mask[..., None]
        feature_mask = frame_mask.expand_as(normalized_target)
        if inputs.body_feature_valid_mask is not None:
            feature_mask = feature_mask & inputs.body_feature_valid_mask[..., :BODY_CONTINUOUS_DIM]
        boundaries = (
            ("position", 0, BODY_POSITION_DIM, self.lambda_position),
            ("rotation", BODY_POSITION_DIM, BODY_POSITION_DIM + BODY_ROTATION_DIM, self.lambda_rotation),
            ("velocity", BODY_POSITION_DIM + BODY_ROTATION_DIM, BODY_CONTINUOUS_DIM, self.lambda_velocity),
        )
        losses: dict[str, torch.Tensor] = {}
        reconstruction = predicted.new_zeros(())
        for name, start, end, weight in boundaries:
            loss = _masked_mean(
                F.smooth_l1_loss(
                    normalized_pred[..., start:end], normalized_target[..., start:end], reduction="none"
                ),
                feature_mask[..., start:end],
            )
            losses[name] = loss
            reconstruction = reconstruction + float(weight) * loss
        contacts = inputs.body_motion[..., BODY_CONTINUOUS_DIM:]
        contact_mask = frame_mask.expand_as(contacts)
        if inputs.body_feature_valid_mask is not None:
            contact_mask = (
                contact_mask
                & inputs.body_feature_valid_mask[..., BODY_CONTINUOUS_DIM:]
            )
        contact_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(
                prediction.body.contact_logits, contacts, reduction="none"
            ),
            contact_mask,
        )
        losses["contact"] = contact_loss
        reconstruction = reconstruction + self.lambda_contact * contact_loss

        direct_non_root = predicted[..., :BODY_POSITION_DIM].reshape(
            *predicted.shape[:2], NUM_JOINTS - 1, 3
        )
        direct_positions = self._global_positions(
            inputs.root_motion, direct_non_root
        )
        foot_indices = torch.as_tensor(self.foot_joint_indices, device=predicted.device)
        foot_positions = direct_positions.index_select(-2, foot_indices)
        foot_speed = predicted.new_zeros(*predicted.shape[:2], len(self.foot_joint_indices))
        foot_speed[:, 1:] = (
            foot_positions[:, 1:] - foot_positions[:, :-1]
        ).norm(dim=-1) * self.fps
        non_root_foot_indices = foot_indices - 1
        foot_position_valid = feature_mask[..., :BODY_POSITION_DIM].reshape(
            *predicted.shape[:2], NUM_JOINTS - 1, 3
        ).all(dim=-1).index_select(-1, non_root_foot_indices)
        transition_valid = torch.zeros_like(foot_position_valid)
        transition_valid[:, 1:] = (
            foot_position_valid[:, 1:]
            & foot_position_valid[:, :-1]
            & inputs.frame_valid_mask[:, 1:, None]
            & inputs.frame_valid_mask[:, :-1, None]
        )
        skating_mask = contact_mask & transition_valid
        skating = _masked_mean(
            contacts * foot_speed,
            skating_mask,
        )
        losses["skating"] = skating

        velocity_start = BODY_POSITION_DIM + BODY_ROTATION_DIM
        velocities = predicted[..., velocity_start:].reshape(
            *predicted.shape[:2], NUM_JOINTS, 3
        )

        if self.lambda_geodesic:
            pred_rot = rotation_to_matrix(
                predicted[..., BODY_POSITION_DIM : BODY_POSITION_DIM + BODY_ROTATION_DIM]
                .reshape(*predicted.shape[:2], NUM_JOINTS, 6)
            )
            target_rot = rotation_to_matrix(
                target[..., BODY_POSITION_DIM : BODY_POSITION_DIM + BODY_ROTATION_DIM]
                .reshape(*target.shape[:2], NUM_JOINTS, 6)
            )
            relative = pred_rot.transpose(-1, -2) @ target_rot
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1 + 1e-6, 1 - 1e-6)
            geodesic = _masked_mean(torch.acos(cosine), frame_mask.expand_as(cosine))
        else:
            geodesic = predicted.new_zeros(())
        losses["geodesic"] = geodesic

        if self.lambda_fk or self.lambda_position_consistency:
            pred_rot = rotation_to_matrix(
                predicted[..., BODY_POSITION_DIM : BODY_POSITION_DIM + BODY_ROTATION_DIM]
                .reshape(*predicted.shape[:2], NUM_JOINTS, 6)
            )
            fk_positions = self._forward_kinematics(inputs.root_motion, pred_rot)
            target_positions = self._global_positions(
                inputs.root_motion,
                target[..., :BODY_POSITION_DIM].reshape(*target.shape[:2], NUM_JOINTS - 1, 3),
            )
            joint_mask = inputs.frame_valid_mask[..., None, None].expand_as(fk_positions)
            fk_loss = _masked_mean(F.smooth_l1_loss(fk_positions, target_positions, reduction="none"), joint_mask)
            position_consistency = _masked_mean(
                F.smooth_l1_loss(fk_positions, direct_positions, reduction="none"), joint_mask
            )
        else:
            fk_loss = position_consistency = predicted.new_zeros(())
        losses["fk"] = fk_loss
        losses["position_consistency"] = position_consistency

        if self.lambda_velocity_consistency:
            derived_velocity = torch.zeros_like(velocities)
            derived_velocity[:, 1:] = (
                direct_positions[:, 1:] - direct_positions[:, :-1]
            ) * self.fps
            velocity_valid = inputs.frame_valid_mask.clone()
            velocity_valid[:, 0] = False
            velocity_valid[:, 1:] &= inputs.frame_valid_mask[:, :-1]
            velocity_consistency = _masked_mean(
                F.smooth_l1_loss(velocities, derived_velocity, reduction="none"),
                velocity_valid[..., None, None].expand_as(velocities),
            )
        else:
            velocity_consistency = predicted.new_zeros(())
        losses["velocity_consistency"] = velocity_consistency

        token_mask = frame_valid_to_token_valid(inputs.frame_valid_mask)
        kl_element = -0.5 * (
            1 + prediction.posterior.logvar - prediction.posterior.mu.square()
            - prediction.posterior.logvar.exp()
        )
        kl = _masked_mean(kl_element, token_mask[..., None].expand_as(kl_element))
        warmup = 1.0 if self.kl_warmup_steps <= 0 else min(
            1.0, max(0.0, float(global_step) / float(self.kl_warmup_steps))
        )
        effective_beta = self.beta_kl * warmup
        losses["kl"] = kl
        losses["kl_beta"] = kl.new_tensor(effective_beta)
        posterior_sigma = (0.5 * prediction.posterior.logvar).exp()
        losses["posterior_sigma_mean"] = _masked_mean(
            posterior_sigma, token_mask[..., None].expand_as(posterior_sigma)
        )
        losses["posterior_mu_rms"] = _masked_mean(
            prediction.posterior.mu.square(),
            token_mask[..., None].expand_as(prediction.posterior.mu),
        ).sqrt()
        flattened_mu = prediction.posterior.mu[token_mask]
        if flattened_mu.shape[0] > 1:
            active = flattened_mu.var(dim=0, unbiased=False) > 1e-2
            losses["active_latent_fraction"] = active.float().mean()
        else:
            losses["active_latent_fraction"] = kl.new_zeros(())
        losses["reconstruction"] = reconstruction
        losses["total"] = (
            reconstruction
            + self.lambda_skating * skating
            + effective_beta * kl
            + self.lambda_geodesic * geodesic
            + self.lambda_fk * fk_loss
            + self.lambda_position_consistency * position_consistency
            + self.lambda_velocity_consistency * velocity_consistency
        )
        return losses


__all__ = ["VAELoss"]
