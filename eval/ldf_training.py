"""Entrypoint-level composition for optional LDF generation evaluation."""

from __future__ import annotations

from lightning.pytorch.callbacks import Callback

from utils.training.ldf.evaluation.runner import LDFEvaluationRunner


class LDFEvaluationCallback(Callback):
    """Run heavyweight generation probes without coupling them to training core."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.runner = LDFEvaluationRunner(cfg)

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self.runner.validate_text_coverage(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        # Callback hooks run before the module hook. BasicLightningModule has
        # already installed EMA weights in on_validation_start and restores
        # them in its own on_validation_epoch_end after this callback returns.
        self.runner.maybe_run(pl_module)


__all__ = ["LDFEvaluationCallback"]
