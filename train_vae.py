"""Train the strict-4 body VAE from native-rotation artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils.conditions.vae import VAEInput
from utils.initialize import get_function, get_shared_run_time, instantiate, load_config, save_config_and_codes
from utils.training.lightning_module import BasicLightningModule
from utils.training.vae_loss import VAELoss


class Strict4VAELightningModule(BasicLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_fn = VAELoss(
            body_cont_mean=self.model.body_cont_mean,
            body_cont_std=self.model.body_cont_std,
            **dict(cfg.loss),
        )

    @staticmethod
    def _inputs(batch) -> VAEInput:
        return VAEInput(
            body_motion=batch["body_motion"],
            root_motion=batch["root_motion"],
            frame_valid_mask=batch["frame_valid_mask"],
            previous_root_frame=batch.get("previous_root_frame"),
            previous_root_valid_mask=batch.get("previous_root_valid_mask"),
            body_feature_valid_mask=batch.get("body_feature_valid_mask"),
        )

    def _step(self, batch, is_training=True):
        inputs = self._inputs(batch)
        prediction = self.model(inputs)
        return self.loss_fn(inputs, prediction, global_step=int(self.global_step))


def _dataset(cfg, split: str):
    if not cfg.data.manifest_path:
        raise RuntimeError(
            "STRICT4_NATIVE_ROTATIONS_REQUIRED: set data.manifest_path or STRICT4_MANIFEST "
            "to a manifest produced from native SMPL/AMASS rotations. Legacy263 and IK "
            "fallbacks are intentionally unsupported."
        )
    return instantiate(
        cfg.data.target,
        cfg=None,
        manifest_path=cfg.data.manifest_path,
        split=split,
        min_frames=cfg.data.min_frames,
        max_frames=cfg.data.max_frames,
        random_yaw=cfg.data.random_yaw,
    )


def main() -> None:
    torch.set_float32_matmul_precision("high")
    cfg = load_config()
    seed_everything(cfg.seed)
    if not cfg.model.params.motion_stats_path and not cfg.model.params.get("allow_identity_statistics", False):
        raise RuntimeError(
            "STRICT4_MOTION_STATS_REQUIRED: compute train-split statistics before real VAE training"
        )
    train_dataset = _dataset(cfg, "train") if cfg.train else None
    val_dataset = _dataset(cfg, "val")
    collate = get_function(cfg.data.collate_fn)
    loader_kwargs = dict(num_workers=cfg.data.num_workers, collate_fn=collate)
    train_loader = (
        DataLoader(train_dataset, batch_size=cfg.data.train_bs, shuffle=True, **loader_kwargs)
        if train_dataset is not None else None
    )
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.val_bs, shuffle=False, **loader_kwargs)

    run_time = get_shared_run_time(cfg.save_dir)
    save_dir = Path(cfg.save_dir) / f"{run_time}_{cfg.exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.update(cfg.config, "save_dir", str(save_dir))
    save_config_and_codes(cfg, str(save_dir))
    logger = None
    if not cfg.debug and cfg.get("logger") and cfg.logger.wandb.wandb_key:
        os.environ["WANDB_API_KEY"] = cfg.logger.wandb.wandb_key
        logger = WandbLogger(
            project=cfg.logger.wandb.project,
            entity=cfg.logger.wandb.entity,
            name=f"{cfg.exp_name}_{run_time}",
            save_dir=str(save_dir),
        )
    module = Strict4VAELightningModule(cfg.config)
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="step_{step}",
        every_n_train_steps=cfg.validation.save_every_n_steps,
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    trainer = Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint],
        default_root_dir=str(save_dir),
        val_check_interval=cfg.validation.validation_steps,
        check_val_every_n_epoch=None,
    )
    if cfg.train:
        trainer.fit(module, train_loader, val_dataloaders=val_loader, ckpt_path=cfg.resume_ckpt)
    else:
        trainer.validate(module, dataloaders=val_loader, ckpt_path=cfg.test_ckpt)


if __name__ == "__main__":
    main()
