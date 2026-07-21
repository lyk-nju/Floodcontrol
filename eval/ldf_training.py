"""Entrypoint-level composition for optional LDF generation evaluation."""

from __future__ import annotations

from lightning.pytorch.callbacks import Callback

from utils.training.ldf.evaluation.runner import LDFEvaluationRunner


class LDFEvaluationCallback(Callback):
    """Run heavyweight generation probes without coupling them to training core."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.runner = LDFEvaluationRunner(cfg)

    def on_validation_start(self, trainer, pl_module) -> None:
        # Every rank validates its local lookup before entering generation
        # collectives.  A rank-local failure must stop the whole DDP job rather
        # than leave peers waiting in a later all-gather.
        self.runner.validate_text_coverage(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        # Callback hooks run before the module hook. BasicLightningModule has
        # already installed EMA weights in on_validation_start and restores
        # them in its own on_validation_epoch_end after this callback returns.
        # All ranks participate.  The runner shards generation work and keeps
        # summaries/logger writes rank-zero-only.
        if self.runner.run_at_start(pl_module):
            return
        self.runner.maybe_run(pl_module)


__all__ = ["LDFEvaluationCallback"]
