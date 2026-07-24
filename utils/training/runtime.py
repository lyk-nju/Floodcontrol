"""Shared construction helpers for Lightning training entrypoints."""

from __future__ import annotations

import os
from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf

from utils.initialize import get_shared_run_timestamp, save_run_snapshot


def _process_rank() -> int:
    return int(
        os.environ.get(
            "RANK",
            os.environ.get("GLOBAL_RANK", os.environ.get("LOCAL_RANK", "0")),
        )
    )


def create_run_directory(cfg) -> tuple[str, Path]:
    """Create one shared run directory and persist its resolved snapshot."""

    run_time = get_shared_run_timestamp(cfg.save_dir)
    save_dir = Path(cfg.save_dir) / f"{run_time}_{cfg.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", str(save_dir))
    OmegaConf.update(cfg.config, "run_time", run_time)
    rank_zero_info(f"Save dir: {save_dir}, exp_name: {cfg.exp_name}")
    if _process_rank() == 0:
        save_run_snapshot(cfg, save_dir)
    return run_time, save_dir


def create_wandb_logger(cfg, run_time: str, save_dir: Path):
    """Create the project WandB logger when the resolved run enables it."""

    if cfg.debug or not cfg.get("wandb_info"):
        return None
    key = cfg.wandb_info.get("key")
    if key is None or not str(key).strip():
        rank_zero_info("WandB API key not provided, skipping WandB logging")
        return None
    os.environ["WANDB_API_KEY"] = str(key)
    rank_zero_info("WandB logging enabled")
    return WandbLogger(
        project=cfg.wandb_info.project,
        entity=cfg.wandb_info.entity,
        name=f"{cfg.exp_name}_{run_time}",
        config=OmegaConf.to_container(cfg.config, resolve=True),
        save_dir=str(save_dir),
    )


def create_step_checkpoint_callback(cfg, save_dir: Path) -> ModelCheckpoint:
    """Create the common absolute-step checkpoint policy."""

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
