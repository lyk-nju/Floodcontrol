"""Train or validate the Floodcontrol body VAE."""

from __future__ import annotations

import torch
from lightning import Trainer, seed_everything

from utils.initialize import load_config
from utils.training.runtime import (
    create_run_directory,
    create_step_checkpoint_callback,
    create_wandb_logger,
)
from utils.training.vae.data import create_dataloaders
from utils.training.vae.lightning_module import VAELightningModule
from utils.training.lightning_module import EMARestoreOnException


def _validate_training_config(cfg) -> None:
    if not cfg.model.params.get("motion_stats_path"):
        raise RuntimeError(
            "MOTION_STATISTICS_REQUIRED: compute train-split statistics before real VAE training"
        )


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    _validate_training_config(cfg)

    train_dataloader, val_dataloader = create_dataloaders(cfg)
    run_time, save_dir = create_run_directory(cfg)
    logger = create_wandb_logger(cfg, run_time, save_dir)
    lightning_module = VAELightningModule(cfg.config)
    callbacks = [EMARestoreOnException()]
    if cfg.train:
        callbacks.append(create_step_checkpoint_callback(cfg, save_dir))
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(save_dir),
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    if cfg.train:
        trainer.fit(
            lightning_module,
            train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.resume_ckpt,
            # Project-generated training checkpoints contain trusted optimizer,
            # scheduler and OmegaConf state in addition to tensors.
            weights_only=False,
        )
    else:
        trainer.validate(
            lightning_module,
            dataloaders=val_dataloader,
            ckpt_path=cfg.test_ckpt,
            weights_only=False,
        )


if __name__ == "__main__":
    main()
