"""Train or validate the hybrid root/body latent diffusion forcing model."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf

from utils.initialize import get_shared_run_timestamp, load_config, save_run_snapshot
from utils.training.ldf import LDFLightningModule, create_dataloaders


def _require_file(path: str, name: str) -> None:
    if not Path(path).is_file():
        raise RuntimeError(f"{name}_REQUIRED: file not found at {path}")


def _validate_training_config(cfg) -> None:
    if str(cfg.status) != "training_ready":
        raise RuntimeError(
            "LDF_ROOT_STATISTICS_REQUIRED: regenerate root_stats.npz with the "
            "current fixed-span/anchor sampler, then explicitly set "
            "status: training_ready"
        )
    _require_file(str(cfg.root_stats_path), "ROOT_STATISTICS")
    _require_file(str(cfg.vae.checkpoint_path), "VAE_CHECKPOINT")
    _require_file(str(cfg.vae.params.motion_stats_path), "MOTION_STATISTICS")
    _require_file(str(cfg.vae.params.latent_stats_path), "LATENT_STATISTICS")
    _require_file(str(cfg.text_embeddings_path), "TEXT_EMBEDDINGS")
    if int(cfg.model.params.text_len) != int(cfg.text_encoder.text_len):
        raise ValueError("model.text_len and text_encoder.text_len must match")


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
    key = str(cfg.wandb_info.key)
    if not key.strip():
        return None
    os.environ["WANDB_API_KEY"] = key
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

    module = LDFLightningModule(cfg.config)
    train_loader, val_loader = create_dataloaders(
        cfg,
        encoder_context_tokens=module.vae.encoder_context_tokens,
    )
    run_time, save_dir = _create_run_directory(cfg)
    logger = _create_logger(cfg, run_time, save_dir)
    callbacks = [_create_checkpoint_callback(cfg, save_dir)] if cfg.train else []
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
            module,
            train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.resume_ckpt,
            weights_only=False,
        )
    else:
        trainer.validate(
            module,
            dataloaders=val_loader,
            ckpt_path=cfg.test_ckpt,
            weights_only=False,
        )


if __name__ == "__main__":
    main()
