"""Train or validate the Floodcontrol body VAE."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf

from utils.initialize import (
    get_shared_run_timestamp,
    load_config,
    save_run_snapshot,
)
from utils.training.vae.data import create_dataloaders
from utils.training.vae.lightning_module import VAELightningModule
from utils.training.lightning_module import EMARestoreOnException


def _validate_training_config(cfg) -> None:
    if not cfg.model.params.get("motion_stats_path"):
        raise RuntimeError(
            "MOTION_STATISTICS_REQUIRED: compute train-split statistics before real VAE training"
        )


def _create_run_directory(cfg) -> tuple[str, Path]:
    run_time = get_shared_run_timestamp(cfg.save_dir)
    save_dir = Path(cfg.save_dir) / f"{run_time}_{cfg.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", str(save_dir))
    OmegaConf.update(cfg.config, "run_time", run_time)
    rank_zero_info(f"Save dir: {save_dir}, exp_name: {cfg.exp_name}")
    save_run_snapshot(cfg, save_dir)
    return run_time, save_dir


def _create_logger(cfg, run_time: str, save_dir: Path):
    if cfg.debug or not cfg.get("wandb_info"):
        return None
    wandb_key = cfg.wandb_info.key
    if not wandb_key or not wandb_key.strip():
        rank_zero_info("WandB API key not provided, skipping WandB logging")
        return None
    os.environ["WANDB_API_KEY"] = wandb_key
    rank_zero_info("WandB logging enabled")
    return WandbLogger(
        project=cfg.wandb_info.project,
        entity=cfg.wandb_info.entity,
        name=f"{cfg.exp_name}_{run_time}",
        config=OmegaConf.to_container(cfg.config, resolve=True),
        save_dir=str(save_dir),
    )


def _create_checkpoint_callback(cfg, save_dir: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=save_dir,
        filename="step_{ckpt_absolute_step:06.0f}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=cfg.validation.save_top_k,
        monitor="ckpt_absolute_step",
        mode="max",
        auto_insert_metric_name=False,
        save_last=True,
        save_on_train_epoch_end=False,
    )


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    _validate_training_config(cfg)

    train_dataloader, val_dataloader = create_dataloaders(cfg)
    run_time, save_dir = _create_run_directory(cfg)
    logger = _create_logger(cfg, run_time, save_dir)
    lightning_module = VAELightningModule(cfg.config)
    callbacks = [EMARestoreOnException()]
    if cfg.train:
        callbacks.append(_create_checkpoint_callback(cfg, save_dir))
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
