"""Train or validate the hybrid root/body latent diffusion forcing model."""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf

from utils.initialize import get_shared_run_timestamp, load_config, save_run_snapshot
from utils.training.ldf.data import create_dataloaders
from utils.training.ldf.lightning_module import LDFLightningModule
from utils.training.ldf.self_forcing import validate_self_forcing_config
from utils.training.lightning_module import EMARestoreOnException
from utils.training.memory import CUDAMemoryReporter


def _require_file(path: str, name: str) -> None:
    if not Path(path).is_file():
        raise RuntimeError(f"{name}_REQUIRED: file not found at {path}")


def _validate_training_config(cfg) -> None:
    _require_file(str(cfg.data.root_stats_path), "ROOT_STATISTICS")
    _require_file(str(cfg.vae.checkpoint_path), "VAE_CHECKPOINT")
    _require_file(str(cfg.vae.params.motion_stats_path), "MOTION_STATISTICS")
    _require_file(str(cfg.vae.params.latent_stats_path), "LATENT_STATISTICS")
    _require_file(str(cfg.data.text_embeddings_path), "TEXT_EMBEDDINGS")
    if int(cfg.model.params.text_len) != int(cfg.text_encoder.text_len):
        raise ValueError("model.text_len and text_encoder.text_len must match")
    training = cfg.get("training") or {}
    for name in (
        "text_dropout",
        "constraint_dropout",
    ):
        probability = float(training.get(name, 0.0))
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"training.{name} must lie in [0,1]")
    lookahead = int(training.get("max_horizon_token", 0))
    window = training.get("window") or {}
    max_window_tokens = int(window.get("max_tokens", 0))
    generation_tokens = int(window.get("generation_tokens", 0))
    if max_window_tokens <= 0:
        raise ValueError("training.window.max_tokens must be positive")
    if generation_tokens != int(cfg.model.params.chunk_size):
        raise ValueError(
            "training.window.generation_tokens must equal model.params.chunk_size"
        )
    if str(window.get("sampling", "")) != "random_generation_start":
        raise ValueError(
            "training.window.sampling must be random_generation_start"
        )
    if int(cfg.data.max_frames) != max_window_tokens * 4:
        raise ValueError(
            "data.max_frames must equal 4 * training.window.max_tokens"
        )
    if int(cfg.data.min_frames) < generation_tokens * 4:
        raise ValueError(
            "data.min_frames must contain at least one complete generation chunk"
        )
    if lookahead <= 0 or lookahead > max_window_tokens - generation_tokens:
        raise RuntimeError(
            "LDF_XZ_CONSTRAINT_REQUIRED: "
            "training.max_horizon_token must lie in "
            "[1, window.max_tokens - window.generation_tokens]"
        )
    self_forcing = cfg.get("self_forcing") or {}
    validate_self_forcing_config(
        self_forcing,
        generation_tokens=generation_tokens,
        max_window_tokens=max_window_tokens,
        max_steps=int(cfg.trainer.max_steps),
    )
    if bool(self_forcing.get("enabled", False)) and (
        int(cfg.model.params.noise_steps) % generation_tokens
    ):
        raise ValueError(
            "persistent self-forcing requires model.noise_steps divisible by "
            "training.window.generation_tokens"
        )
    loss = cfg.get("loss") or {}
    for name in (
        "root_weight",
        "body_weight",
        "rollout_weight",
        "root_boundary_weight",
    ):
        value = float(loss.get(name, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"loss.{name} must be finite and non-negative")
    beta_min = float(loss.get("offpath_beta_min", 0.1))
    if not math.isfinite(beta_min) or beta_min <= 0.0:
        raise ValueError("loss.offpath_beta_min must be finite and positive")
    sampling = training.get("constraint_sampling") or {}
    probabilities = [
        float(sampling.get(name, -1.0))
        for name in (
            "dense_probability",
            "waypoint_probability",
            "goal_probability",
        )
    ]
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError(
            "training.constraint_sampling probabilities must lie in [0,1]"
        )
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError(
            "training.constraint_sampling probabilities must sum to one"
        )
    if int(sampling.get("max_waypoint_count", 0)) <= 0:
        raise ValueError(
            "training.constraint_sampling.max_waypoint_count must be positive"
        )

    validation = cfg.get("validation") or {}
    generation = validation.get("generation") or {}
    dense_xz = validation.get("dense_xz") or {}
    t2m = validation.get("t2m") or {}
    generation_enabled = bool(generation.get("enabled", False))
    if generation_enabled:
        validation_steps = int(validation.validation_steps)
        generation_steps = int(generation.get("steps", 0))
        if generation_steps <= 0 or generation_steps % validation_steps:
            raise ValueError(
                "validation.generation.steps must be a positive multiple of "
                "validation.validation_steps"
            )
        modes = [str(mode) for mode in generation.get("modes", [])]
        if not modes or len(set(modes)) != len(modes) or not set(modes) <= {
            "stream",
            "rolling",
        }:
            raise ValueError(
                "validation.generation.modes must be unique stream/rolling values"
            )
        for name in ("num_runs", "num_denoise_steps", "max_horizon_token"):
            if int(generation.get(name, 0)) <= 0:
                raise ValueError(f"validation.generation.{name} must be positive")
        rolling = generation.get("rolling") or {}
        rolling_tokens = int(rolling.get("window_tokens", 0))
        if rolling_tokens <= generation_tokens:
            raise ValueError(
                "validation.generation.rolling.window_tokens must exceed chunk_size"
            )
        if rolling_tokens > max_window_tokens:
            raise ValueError(
                "validation rolling window cannot exceed training.window.max_tokens"
            )
        if int(generation.get("max_horizon_token", 0)) > lookahead:
            raise ValueError(
                "validation future constraints cannot exceed the training lookahead cap"
            )
        if bool(dense_xz.get("enabled", False)):
            if int(dense_xz.get("segment_frames", 0)) <= 0:
                raise ValueError("validation.dense_xz.segment_frames must be positive")
            video_samples = int(dense_xz.get("video_samples", 0))
            if video_samples < 0:
                raise ValueError("validation.dense_xz.video_samples must be non-negative")
            probe = str(dense_xz.get("probe", "")).strip()
            if not probe:
                raise ValueError("validation.dense_xz.probe must name a data probe")
            probe_paths = cfg.data.get("test_probe_meta_paths") or {}
            if probe not in probe_paths or not probe_paths[probe]:
                raise ValueError(
                    f"data.test_probe_meta_paths must define {probe!r}"
                )
            for index, path in enumerate(probe_paths[probe]):
                _require_file(str(path), f"DENSE_XZ_PROBE_{index}")
        if bool(t2m.get("enabled", False)):
            t2m_steps = int(t2m.get("steps", 0))
            if t2m_steps <= 0 or t2m_steps % validation_steps:
                raise ValueError(
                    "validation.t2m.steps must be a positive multiple of "
                    "validation.validation_steps"
                )
            if cfg.get("metrics") is None or cfg.metrics.get("t2m") is None:
                raise ValueError("validation.t2m requires metrics.t2m")
            metric_cfg = cfg.metrics.t2m
            for name, path in (
                ("T2M_MEAN", metric_cfg.metric_mean_path),
                ("T2M_STD", metric_cfg.metric_std_path),
                ("T2M_TEXT_ENCODER", metric_cfg.textencoder.ckpt),
                ("T2M_MOVEMENT_ENCODER", metric_cfg.moveencoder.ckpt),
                ("T2M_MOTION_ENCODER", metric_cfg.motionencoder.ckpt),
            ):
                _require_file(str(path), name)


def _create_run_directory(cfg) -> tuple[str, Path]:
    run_time = get_shared_run_timestamp(cfg.save_dir)
    save_dir = Path(cfg.save_dir) / f"{run_time}_{cfg.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", str(save_dir))
    OmegaConf.update(cfg.config, "run_time", run_time)
    rank_zero_info(f"Save dir: {save_dir}, exp_name: {cfg.exp_name}")
    process_rank = int(
        os.environ.get(
            "RANK",
            os.environ.get("GLOBAL_RANK", os.environ.get("LOCAL_RANK", "0")),
        )
    )
    if process_rank == 0:
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


def _generation_evaluation_enabled(cfg) -> bool:
    validation = cfg.get("validation") or {}
    generation = validation.get("generation") or {}
    dense_xz = validation.get("dense_xz") or {}
    t2m = validation.get("t2m") or {}
    return bool(generation.get("enabled", False)) and (
        bool(dense_xz.get("enabled", False)) or bool(t2m.get("enabled", False))
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
    callbacks = [EMARestoreOnException()]
    if _generation_evaluation_enabled(cfg):
        # Keep metrics/video/inference dependencies outside the LDF training
        # kernel and load them only for runs that request full generation eval.
        from eval.ldf_training import LDFEvaluationCallback

        callbacks.append(LDFEvaluationCallback(cfg.config))
    # Keep this after generation evaluation so validation-time inference peaks
    # are visible to the reporter before it samples the cumulative CUDA peak.
    callbacks.append(CUDAMemoryReporter(cfg.config))
    if cfg.train:
        callbacks.append(_create_checkpoint_callback(cfg, save_dir))
    trainer_config = OmegaConf.to_container(cfg.trainer, resolve=True)
    # LDF owns length-bucket training sharding and exact validation sharding.
    # Letting Lightning inject another DistributedSampler would either
    # duplicate sharding or fail on the resumable batch sampler.
    trainer_config["use_distributed_sampler"] = False
    trainer = Trainer(
        **trainer_config,
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
