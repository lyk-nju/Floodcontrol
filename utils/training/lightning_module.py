import time

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import rank_zero_info
from torch.nn.modules.module import _IncompatibleKeys
from torch_ema import ExponentialMovingAverage

from utils.initialize import (
    instantiate_target,
    log_model_parameters,
    log_state_dict_summary,
)
from utils.training.module_step import ckpt_step_info


class BasicLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate_target(
            target=cfg.model.target,
            cfg=None,
            hfstyle=False,
            **cfg.model.params,
        )

        self._trainable_parameters = tuple(
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        )
        self.ema = ExponentialMovingAverage(
            self._trainable_parameters,
            decay=cfg.model.ema_decay,
        )
        log_model_parameters(self.model)

        self.last_batch_end_time, self.batch_ready_time = None, None
        self._skip_next_lightning_load_state_dict = False

    def configure_optimizers(self):
        optim_target = self.cfg.optimizer.target
        if len(optim_target.split(".")) == 1:
            optim_target = "torch.optim." + optim_target
        optimizer = instantiate_target(
            target=optim_target,
            cfg=None,
            hfstyle=False,
            params=self._trainable_parameters,
            **self.cfg.optimizer.params,
        )

        scheduler_target = self.cfg.lr_scheduler.target
        if len(scheduler_target.split(".")) == 1:
            scheduler_target = "torch.optim.lr_scheduler." + scheduler_target
        lr_scheduler = instantiate_target(
            target=scheduler_target,
            cfg=None,
            hfstyle=False,
            optimizer=optimizer,
            **self.cfg.lr_scheduler.params,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        if self._skip_next_lightning_load_state_dict:
            self._skip_next_lightning_load_state_dict = False
            return _IncompatibleKeys([], [])
        return self.model.load_state_dict(state_dict, strict=strict)

    def on_load_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
            rank_zero_info("init ema from ckpt")
        else:
            self.ema = ExponentialMovingAverage(
                self._trainable_parameters,
                decay=self.cfg.model.ema_decay,
            )
            rank_zero_info("init ema from current model weights")

        # Compare state_dict and parameters
        log_state_dict_summary(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )
        self._skip_next_lightning_load_state_dict = True

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema_state"] = self.ema.state_dict()
        checkpoint["state_dict"] = self.model.state_dict()

    def _step(self, batch, is_training=True):
        raise NotImplementedError

    def on_fit_start(self):
        self.ema.to(self.device)

    def on_validation_start(self):
        self.ema.to(self.device)

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_ready_time = time.time()

    def training_step(self, batch, batch_idx):
        net_start_time = time.time()
        loss_dict = self._step(batch, is_training=True)
        net_end_time = time.time()
        data_time = (
            self.batch_ready_time - self.last_batch_end_time
            if self.last_batch_end_time is not None
            else 0.0
        )
        net_time = net_end_time - net_start_time
        batch_size = int(batch["body_motion"].shape[0])
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "data_time", data_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "net_time", net_time, on_step=True, prog_bar=True, batch_size=batch_size
        )
        self.log(
            "ckpt_absolute_step",
            float(ckpt_step_info(self).metric_value),
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for key, value in loss_dict.items():
            self.log(
                f"train_loss/{key}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss_dict["total"]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.last_batch_end_time = time.time()
        self.ema.update()
        if self.global_step % 100 == 0:
            self.log("ema_decay", self.ema.decay, sync_dist=False)
            with torch.no_grad():
                total_abs, total_n = torch.zeros((), device=self.device), 0
                for parameter, shadow in zip(
                    self._trainable_parameters, self.ema.shadow_params, strict=True
                ):
                    total_abs += (parameter.detach() - shadow).abs().sum()
                    total_n += parameter.numel()
                avg_diff = total_abs / max(total_n, 1)
                self.log("ema_diff/avg", avg_diff, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        with self.ema.average_parameters(self._trainable_parameters):
            loss_dict = self._step(batch, is_training=False)
        batch_size = int(batch["body_motion"].shape[0])
        for key, value in loss_dict.items():
            self.log(
                f"val_loss/{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
