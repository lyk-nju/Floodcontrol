"""Lightning integration for body VAE optimization."""

from __future__ import annotations

import torch

from utils.conditions.vae import VAEPrediction, VAEInput
from utils.training.lightning_module import BasicLightningModule

from .checkpoint import PHYSICAL_STATISTIC_BUFFERS
from .losses import VAELoss
from .metrics import reconstruction_geometry_metrics


def validate_resume_checkpoint(
    model,
    checkpoint,
) -> None:
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("VAE resume checkpoint is missing state_dict")
    configured = model.state_dict()
    for name in PHYSICAL_STATISTIC_BUFFERS:
        saved = state.get(name)
        if not torch.is_tensor(saved) or not torch.equal(
            saved.detach().cpu(), configured[name].detach().cpu()
        ):
            raise RuntimeError(
                f"VAE_RESUME_STATISTICS_MISMATCH: checkpoint buffer {name!r} "
                "does not match the configured motion statistics"
            )


class VAELightningModule(BasicLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_fn = VAELoss(
            body_cont_mean=self.model.body_cont_mean,
            body_cont_std=self.model.body_cont_std,
            **dict(cfg.loss),
        )

    def on_load_checkpoint(self, checkpoint) -> None:
        validate_resume_checkpoint(self.model, checkpoint)
        # Existing training checkpoints stored placeholder latent statistics.
        # They are not model state anymore and must not enter resumed training.
        checkpoint["state_dict"].pop("latent_mean", None)
        checkpoint["state_dict"].pop("latent_std", None)
        super().on_load_checkpoint(checkpoint)

    @staticmethod
    def _create_input(batch) -> VAEInput:
        return VAEInput(
            body_motion=batch["body_motion"],
            root_motion=batch["root_motion"],
            frame_valid_mask=batch["frame_valid_mask"],
            previous_root_frame=batch.get("previous_root_frame"),
            previous_root_valid_mask=batch.get("previous_root_valid_mask"),
            body_feature_valid_mask=batch.get("body_feature_valid_mask"),
        )

    def _step(self, batch, is_training=True):
        inputs = self._create_input(batch)
        prediction = self.model(inputs)
        losses = self.loss_fn(
            inputs,
            prediction,
            global_step=int(self.global_step),
        )
        if not is_training:
            mu_body = self.model.decode(
                prediction.posterior.mu,
                prediction.local_root_motion,
                prediction.local_root_valid_mask,
                inputs.frame_valid_mask,
            )
            mu_prediction = VAEPrediction(
                body=mu_body,
                posterior=prediction.posterior,
                latent_sample=prediction.posterior.mu,
                local_root_motion=prediction.local_root_motion,
                local_root_valid_mask=prediction.local_root_valid_mask,
            )
            mu_losses = self.loss_fn(
                inputs,
                mu_prediction,
                global_step=int(self.global_step),
            )
            for name in (
                "position", "rotation", "velocity", "contact", "skating",
                "reconstruction", "total",
            ):
                losses[f"mu_{name}"] = mu_losses[name]
            geometry_metrics = reconstruction_geometry_metrics(
                inputs,
                mu_prediction.body,
            )
            for name, value in geometry_metrics.items():
                losses[f"metric/{name}"] = value
        return losses


__all__ = ["VAELightningModule", "validate_resume_checkpoint"]
