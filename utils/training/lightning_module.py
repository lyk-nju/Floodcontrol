import time
from contextlib import contextmanager

import torch
from lightning import LightningModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info
from torch.nn.modules.module import _IncompatibleKeys
from torch_ema import ExponentialMovingAverage

from utils.initialize import (
    instantiate_target,
    log_model_parameters,
    log_state_dict_summary,
)

_NUMERICAL_CHECK_INTERVAL_STEPS = 500
_TIMING_INTERVAL_STEPS = 50


class BasicLightningModule(LightningModule):
    def __init__(self, cfg, *, model=None):
        super().__init__()
        self.cfg = cfg
        self.model = (
            instantiate_target(
                target=cfg.model.target,
                cfg=None,
                hfstyle=False,
                **cfg.model.params,
            )
            if model is None
            else model
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
        self._timing_active = False
        self._timing_cpu_start = None
        self._timing_cpu_forward_end = None
        self._timing_cuda_start = None
        self._timing_cuda_forward_end = None
        self._detailed_numerical_validation_active = True
        self._numerical_validation_modules_cache = None
        self._skip_next_lightning_load_state_dict = False
        self._ema_parameters_active = False

    @staticmethod
    def _config_get(config, name: str, default=None):
        if config is None:
            return default
        getter = getattr(config, "get", None)
        if callable(getter):
            return getter(name, default)
        return getattr(config, name, default)

    def _detailed_numerical_check_due(
        self, *, step: int, is_training: bool
    ) -> bool:
        if not is_training:
            return True
        if bool(self._config_get(self.cfg, "debug", False)):
            return True
        return int(step) % _NUMERICAL_CHECK_INTERVAL_STEPS == 0

    @contextmanager
    def _numerical_validation_scope(self, *, is_training: bool):
        """Enable expensive tensor-value checks only at fixed periodic steps.

        Shape and dtype checks remain active in every forward. Modules such as
        ``BodyVAE`` opt into this scope through the
        ``numerical_validation_enabled`` attribute. Standalone evaluation and
        inference retain their strict default because the attribute is changed
        only for the duration of a Lightning step.
        """

        enabled = self._detailed_numerical_check_due(
            step=int(self.global_step),
            is_training=is_training,
        )
        if self._numerical_validation_modules_cache is None:
            self._numerical_validation_modules_cache = tuple(
                module
                for module in self.modules()
                if hasattr(module, "numerical_validation_enabled")
            )
        previous = [
            (module, bool(module.numerical_validation_enabled))
            for module in self._numerical_validation_modules_cache
        ]
        for module, _ in previous:
            module.numerical_validation_enabled = enabled
        old_active = self._detailed_numerical_validation_active
        self._detailed_numerical_validation_active = enabled
        try:
            yield
        finally:
            self._detailed_numerical_validation_active = old_active
            for module, value in previous:
                module.numerical_validation_enabled = value

    @staticmethod
    def _validate_final_loss(loss_dict) -> torch.Tensor:
        """Keep one synchronous fail-fast check on the optimized scalar."""

        if not isinstance(loss_dict, dict) or "total" not in loss_dict:
            raise ValueError("_step() must return a dict containing total")
        total = loss_dict["total"]
        if not torch.is_tensor(total) or total.numel() != 1:
            raise ValueError("loss_dict['total'] must be a scalar tensor")
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError("training loss is non-finite")
        return total

    def _timing_due(self) -> bool:
        return bool(self._config_get(self.cfg, "debug", False)) or (
            int(self.global_step) % _TIMING_INTERVAL_STEPS == 0
        )

    def _start_step_timing(self) -> None:
        self._timing_cpu_start = time.perf_counter()
        self._timing_cpu_forward_end = None
        self._timing_cuda_start = None
        self._timing_cuda_forward_end = None
        if self.device.type == "cuda":
            self._timing_cuda_start = torch.cuda.Event(enable_timing=True)
            self._timing_cuda_forward_end = torch.cuda.Event(enable_timing=True)
            self._timing_cuda_start.record()

    def _mark_forward_timing(self) -> None:
        self._timing_cpu_forward_end = time.perf_counter()
        if self._timing_cuda_forward_end is not None:
            self._timing_cuda_forward_end.record()

    def _finish_step_timing(self) -> dict[str, float]:
        if self._timing_cpu_start is None or self._timing_cpu_forward_end is None:
            return {}
        finish = time.perf_counter()
        if self._timing_cuda_start is None:
            return {
                "net_time": self._timing_cpu_forward_end
                - self._timing_cpu_start,
                "step_time": finish - self._timing_cpu_start,
                "step_wall_time": finish - self._timing_cpu_start,
            }

        cuda_step_end = torch.cuda.Event(enable_timing=True)
        cuda_step_end.record()
        # One intentional synchronization at the sparse timing interval gives
        # truthful forward and forward+backward+optimizer GPU durations.
        cuda_step_end.synchronize()
        synchronized_finish = time.perf_counter()
        return {
            "net_time": self._timing_cuda_start.elapsed_time(
                self._timing_cuda_forward_end
            )
            / 1000.0,
            "step_time": self._timing_cuda_start.elapsed_time(cuda_step_end)
            / 1000.0,
            "step_wall_time": synchronized_finish - self._timing_cpu_start,
        }

    def _move_ema_storage(self, device) -> None:
        """Move EMA buffers without retaining inference tensors.

        Lightning validation runs inside ``torch.inference_mode()`` by
        default.  A tensor created by ``ema.to(device)`` in that scope cannot
        later be updated in-place during training.  Disable inference mode for
        the transfer and repair any storage inherited from an older validation
        or checkpoint-loading path.
        """

        with torch.inference_mode(False):
            self.ema.to(device)
            self.ema.shadow_params = [
                value.detach().clone() if torch.is_inference(value) else value
                for value in self.ema.shadow_params
            ]
            if self.ema.collected_params is not None:
                self.ema.collected_params = [
                    value.detach().clone()
                    if torch.is_inference(value)
                    else value
                    for value in self.ema.collected_params
                ]

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
        with torch.inference_mode(False):
            if "ema_state" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema_state"])
                # Older checkpoints may contain the temporary parameter copy made by
                # torch_ema.average_parameters(). It is never part of the EMA model.
                self.ema.collected_params = None
                rank_zero_info("init ema from ckpt")
            else:
                self.ema = ExponentialMovingAverage(
                    self._trainable_parameters,
                    decay=self.cfg.model.ema_decay,
                )
                rank_zero_info("init ema from current model weights")
        self._move_ema_storage(self.device)
        self.ema.collected_params = None
        self._ema_parameters_active = False

        # Compare state_dict and parameters
        log_state_dict_summary(
            state_dict=self.model.state_dict(),
            named_parameters=self.model.named_parameters(),
            named_buffers=self.model.named_buffers(),
        )
        self._skip_next_lightning_load_state_dict = True

    def on_save_checkpoint(self, checkpoint):
        if self._ema_parameters_active:
            raise RuntimeError(
                "cannot save a checkpoint while EMA parameters are active; "
                "restore the training parameters first"
            )
        ema_state = dict(self.ema.state_dict())
        ema_state["collected_params"] = None
        checkpoint["ema_state"] = ema_state
        checkpoint["state_dict"] = self.model.state_dict()

    def _activate_ema_parameters(self) -> None:
        """Swap EMA shadows into the model once for an evaluation scope."""

        if self._ema_parameters_active:
            return
        self._move_ema_storage(self.device)
        with torch.inference_mode(False), torch.no_grad():
            self.ema.store(self._trainable_parameters)
        with torch.no_grad():
            self.ema.copy_to(self._trainable_parameters)
        self._ema_parameters_active = True

    def _restore_training_parameters(self) -> None:
        """Restore trainable weights and release torch_ema's temporary copy."""

        if not self._ema_parameters_active:
            self.ema.collected_params = None
            return
        try:
            with torch.no_grad():
                self.ema.restore(self._trainable_parameters)
        finally:
            self.ema.collected_params = None
            self._ema_parameters_active = False

    @contextmanager
    def use_ema_parameters(self):
        """Use EMA parameters without cloning again inside a nested scope."""

        already_active = self._ema_parameters_active
        if not already_active:
            self._activate_ema_parameters()
        try:
            yield
        finally:
            if not already_active:
                self._restore_training_parameters()

    def _step(self, batch, is_training=True):
        raise NotImplementedError

    def _training_log_name(self, key: str) -> str:
        """Map one loss-dict key to its public training metric name."""

        return f"train_loss/{key}"

    def _training_log_prog_bar(self, key: str) -> bool:
        """Keep the default progress-bar behavior for existing trainers."""

        return True

    def on_fit_start(self):
        self._move_ema_storage(self.device)

    def on_validation_start(self):
        self._activate_ema_parameters()

    def on_train_batch_start(self, batch, batch_idx):
        self.batch_ready_time = time.perf_counter()
        self._timing_active = self._timing_due()

    def training_step(self, batch, batch_idx):
        if self._timing_active:
            self._start_step_timing()
        with self._numerical_validation_scope(is_training=True):
            loss_dict = self._step(batch, is_training=True)
        total = self._validate_final_loss(loss_dict)
        if self._timing_active:
            self._mark_forward_timing()
        batch_size = int(batch["body_motion"].shape[0])
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "ckpt_absolute_step",
            float(self.global_step + 1),
            on_step=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for key, value in loss_dict.items():
            self.log(
                self._training_log_name(key),
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=self._training_log_prog_bar(key),
                sync_dist=True,
                batch_size=batch_size,
            )
        return total

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_size = int(batch["body_motion"].shape[0])
        if self._timing_active:
            timings = self._finish_step_timing()
            timings["data_time"] = (
                self.batch_ready_time - self.last_batch_end_time
                if self.last_batch_end_time is not None
                else 0.0
            )
            for name, value in timings.items():
                self.log(
                    name,
                    value,
                    on_step=True,
                    prog_bar=name in {"data_time", "net_time", "step_time"},
                    batch_size=batch_size,
                )
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
        self.last_batch_end_time = time.perf_counter()
        self._timing_active = False

    def validation_step(self, batch, batch_idx):
        with self._numerical_validation_scope(is_training=False):
            loss_dict = self._step(batch, is_training=False)
        self._validate_final_loss(loss_dict)
        batch_size = int(batch["body_motion"].shape[0])
        for key, value in loss_dict.items():
            if key.startswith("metric/"):
                log_name = f"val_metric/{key.removeprefix('metric/')}"
            else:
                log_name = f"val_loss/{key}"
            self.log(
                log_name,
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        self._restore_training_parameters()


class EMARestoreOnException(Callback):
    """Keep an interrupted validation run from leaving EMA weights installed."""

    def on_exception(self, trainer, pl_module, exception) -> None:
        restore = getattr(pl_module, "_restore_training_parameters", None)
        if restore is not None:
            restore()


__all__ = ["BasicLightningModule", "EMARestoreOnException"]
