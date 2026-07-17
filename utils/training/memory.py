"""Per-device training-memory estimates and observed CUDA peak reporting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_info


_GIB = 1024**3


def _tensor_bytes(values) -> int:
    return sum(int(value.numel()) * int(value.element_size()) for value in values)


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    """Static per-rank memory that exists independently of activations."""

    resident_bytes: int
    gradient_bytes: int
    optimizer_bytes: int
    ema_bytes: int
    ddp_bucket_bytes: int

    @property
    def fixed_lower_bound_bytes(self) -> int:
        return (
            self.resident_bytes
            + self.gradient_bytes
            + self.optimizer_bytes
            + self.ema_bytes
        )

    @property
    def fixed_with_ddp_bucket_bytes(self) -> int:
        return self.fixed_lower_bound_bytes + self.ddp_bucket_bytes


def estimate_training_memory(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    world_size: int,
) -> TrainingMemoryEstimate:
    """Estimate fixed per-rank tensors after the first optimizer step.

    Activations, CUDA context, allocator fragmentation and kernel workspaces are
    deliberately excluded. Adam/AdamW moments are estimated from parameter
    dtype because their lazy state does not exist before the first step.
    """

    parameters = tuple(module.parameters())
    trainable = tuple(value for value in parameters if value.requires_grad)
    resident = _tensor_bytes(parameters) + _tensor_bytes(module.buffers())
    gradients = _tensor_bytes(trainable)

    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        optimizer_bytes = 2 * gradients
    else:
        optimizer_bytes = _tensor_bytes(
            value
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )

    ema = getattr(module, "ema", None)
    shadows = () if ema is None else getattr(ema, "shadow_params", ())
    ema_bytes = _tensor_bytes(shadows)
    # DDP may allocate a gradient-sized communication bucket. Some strategies
    # alias gradients into buckets, so report this as possible overhead rather
    # than pretending it is always resident.
    ddp_bucket = gradients if int(world_size) > 1 else 0
    return TrainingMemoryEstimate(
        resident_bytes=resident,
        gradient_bytes=gradients,
        optimizer_bytes=optimizer_bytes,
        ema_bytes=ema_bytes,
        ddp_bucket_bytes=ddp_bucket,
    )


def _gib(value: int | float) -> float:
    return float(value) / float(_GIB)


class CUDAMemoryReporter(Callback):
    """Report a static startup budget and cumulative observed CUDA peaks."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._first_train_batch_reported = False
        self._reported_reserved_bytes = 0

    @staticmethod
    def _reset_peak(module) -> None:
        if module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(module.device)

    @staticmethod
    def _distributed_peak(module) -> tuple[int, int]:
        peak = torch.tensor(
            [
                torch.cuda.max_memory_allocated(module.device),
                torch.cuda.max_memory_reserved(module.device),
            ],
            device=module.device,
            dtype=torch.int64,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        return int(peak[0].item()), int(peak[1].item())

    def on_fit_start(self, trainer, pl_module) -> None:
        if pl_module.device.type != "cuda":
            rank_zero_info("[LDF memory] CUDA unavailable; GPU memory report skipped")
            return
        if not trainer.optimizers:
            raise RuntimeError("memory reporter requires a configured optimizer")

        estimate = estimate_training_memory(
            pl_module,
            trainer.optimizers[0],
            world_size=int(trainer.world_size),
        )
        total = int(torch.cuda.get_device_properties(pl_module.device).total_memory)
        training = self.cfg.get("training") or {}
        window = training.get("window") or {}
        precision = str(self.cfg.trainer.get("precision", "32-true"))
        fixed = estimate.fixed_with_ddp_bucket_bytes
        headroom = total - fixed
        rank_zero_info(
            "[LDF memory budget per GPU] "
            f"device={torch.cuda.get_device_name(pl_module.device)}, "
            f"capacity={_gib(total):.2f} GiB, precision={precision}, "
            f"train_batch={int(self.cfg.data.train_batch_size)}, "
            f"val_batch={int(self.cfg.data.val_batch_size)}, "
            f"max_tokens={int(window.get('max_tokens', 0))}, "
            f"max_horizon={int(training.get('max_horizon_token', 0))}"
        )
        rank_zero_info(
            "[LDF memory fixed estimate] "
            f"resident={_gib(estimate.resident_bytes):.2f} GiB, "
            f"gradients={_gib(estimate.gradient_bytes):.2f} GiB, "
            f"optimizer={_gib(estimate.optimizer_bytes):.2f} GiB, "
            f"EMA={_gib(estimate.ema_bytes):.2f} GiB, "
            f"possible_DDP_bucket={_gib(estimate.ddp_bucket_bytes):.2f} GiB, "
            f"fixed_total={_gib(fixed):.2f} GiB, "
            f"activation/context/workspace_headroom={_gib(headroom):.2f} GiB"
        )
        rank_zero_info(
            "[LDF memory note] fixed_total is a conservative static budget, not a "
            "hard maximum; the first complete train step will report the measured peak"
        )
        self._reset_peak(pl_module)

    def on_sanity_check_end(self, trainer, pl_module) -> None:
        # Exclude validation-loader warmup from the first complete training-step
        # measurement while keeping later validation/generation peaks cumulative.
        self._reset_peak(pl_module)

    def _report_peak(self, trainer, pl_module, *, label: str, force: bool) -> None:
        if pl_module.device.type != "cuda":
            return
        allocated, reserved = self._distributed_peak(pl_module)
        if not force and reserved <= self._reported_reserved_bytes:
            return
        self._reported_reserved_bytes = max(self._reported_reserved_bytes, reserved)
        rank_zero_info(
            f"[LDF memory observed peak: {label}] "
            f"allocated={_gib(allocated):.2f} GiB, reserved={_gib(reserved):.2f} GiB "
            f"(maximum across {int(trainer.world_size)} rank(s))"
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._first_train_batch_reported:
            return
        self._first_train_batch_reported = True
        self._report_peak(
            trainer,
            pl_module,
            label="first complete train step",
            force=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        self._report_peak(
            trainer,
            pl_module,
            label=f"validation at step {int(pl_module.global_step)}",
            force=False,
        )

    def on_fit_end(self, trainer, pl_module) -> None:
        self._report_peak(trainer, pl_module, label="complete run", force=True)


__all__ = [
    "CUDAMemoryReporter",
    "TrainingMemoryEstimate",
    "estimate_training_memory",
]
