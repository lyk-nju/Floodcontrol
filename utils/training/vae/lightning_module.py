"""Lightning integration for body VAE optimization."""

from __future__ import annotations

from utils.conditions.vae import VAEInput
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
        return self.loss_fn(
            inputs,
            prediction,
            global_step=int(self.global_step),
        )


__all__ = ["VAELightningModule"]
