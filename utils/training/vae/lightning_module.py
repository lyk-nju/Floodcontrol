"""Lightning integration for body VAE optimization."""

from __future__ import annotations

from utils.conditions.vae import VAEPrediction, VAEInput
from utils.training.lightning_module import BasicLightningModule

from .losses import VAELoss


class VAELightningModule(BasicLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_fn = VAELoss(
            body_cont_mean=self.model.body_cont_mean,
            body_cont_std=self.model.body_cont_std,
            **dict(cfg.loss),
        )

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
        return losses


__all__ = ["VAELightningModule"]
