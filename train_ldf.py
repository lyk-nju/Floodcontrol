"""Train or validate the hybrid root/body latent diffusion forcing model."""

from __future__ import annotations

import os

# Lightning's subprocess launcher can import PyTorch before applying its own
# thread heuristic. Keep every rank single-threaded before that first import so
# large DDP jobs do not exhaust the host task limit during NCCL initialization.
for _thread_environment_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_environment_name] = "1"

import torch
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

from utils.initialize import load_config
from utils.training.ldf.config import (
    generation_evaluation_enabled,
    validate_training_config,
)
from utils.training.ldf.data import create_dataloaders
from utils.training.ldf.lightning_module import LDFLightningModule
from utils.training.lightning_module import EMARestoreOnException
from utils.training.memory import CUDAMemoryReporter
from utils.training.runtime import (
    create_run_directory,
    create_step_checkpoint_callback,
    create_wandb_logger,
)


def main() -> None:
    """Build the resolved LDF run and launch training or validation."""

    # Initialization
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    validate_training_config(cfg)

    # Model, data, and run metadata
    module = LDFLightningModule(cfg.config)
    train_loader, val_loader = create_dataloaders(
        cfg,
        encoder_context_tokens=module.vae.encoder_context_tokens,
    )
    run_time, save_dir = create_run_directory(cfg)
    logger = create_wandb_logger(cfg, run_time, save_dir)

    # Callbacks and Trainer
    callbacks = [EMARestoreOnException()]
    if generation_evaluation_enabled(cfg):
        # Metrics, rendering, and inference dependencies remain lazy for
        # training runs that do not request generation evaluation.
        from eval.ldf_training import LDFEvaluationCallback

        callbacks.append(LDFEvaluationCallback(cfg.config))
    # Keep the reporter after generation evaluation so it observes the full
    # validation-time CUDA peak.
    callbacks.append(CUDAMemoryReporter(cfg.config))
    if cfg.train:
        callbacks.append(create_step_checkpoint_callback(cfg, save_dir))

    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    # LDF owns length-bucket training sharding and exact validation sharding.
    # A second DistributedSampler would duplicate that ownership.
    trainer_config["use_distributed_sampler"] = False
    trainer = Trainer(
        **trainer_config,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=str(save_dir),
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )

    # Train or validate
    if cfg.train:
        if bool(cfg.validation.generation.get("run_at_start", False)):
            trainer.validate(
                module,
                dataloaders=val_loader,
                ckpt_path=cfg.resume_ckpt,
                weights_only=False,
            )
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
