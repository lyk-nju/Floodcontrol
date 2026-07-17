"""Lazy public exports for the LDF training package.

Importing a low-level module such as :mod:`utils.training.ldf.flow` must not
load Lightning, generation metrics, video rendering, or inference runtime
code.  Keep the convenient package-level API without turning ``__init__``
into a dependency hub.
"""

from importlib import import_module


__all__ = [
    "LDFLightningModule",
    "LDFSpanCollator",
    "LengthBucketBatchSampler",
    "LDFStepView",
    "LDFTrainingStep",
    "LDFWindowPlan",
    "SelfForcingResult",
    "SelfForcingState",
    "TextEmbeddingLookup",
    "anchor_physical_batch",
    "build_ldf_training_step",
    "create_xz_condition",
    "compute_offpath_loss",
    "compute_velocity_loss",
    "create_dataloaders",
    "create_dataset",
    "resolve_self_forcing_k",
    "run_self_forcing_rollout",
    "sample_rollout_steps",
    "sample_constraint_keep_mask",
    "sample_xz_constraint_mask",
    "sample_window_plan",
    "self_forcing_phase_progress",
    "validate_self_forcing_config",
]


_EXPORTS = {
    "LDFLightningModule": ("utils.training.ldf.lightning_module", "LDFLightningModule"),
    "LDFSpanCollator": ("utils.training.ldf.data", "LDFSpanCollator"),
    "LengthBucketBatchSampler": (
        "utils.training.ldf.data",
        "LengthBucketBatchSampler",
    ),
    "LDFStepView": ("utils.training.ldf.batch", "LDFStepView"),
    "LDFTrainingStep": ("utils.training.ldf.batch", "LDFTrainingStep"),
    "LDFWindowPlan": ("utils.training.ldf.self_forcing", "LDFWindowPlan"),
    "SelfForcingResult": ("utils.training.ldf.self_forcing", "SelfForcingResult"),
    "SelfForcingState": ("utils.training.ldf.self_forcing", "SelfForcingState"),
    "TextEmbeddingLookup": ("utils.training.ldf.text", "TextEmbeddingLookup"),
    "anchor_physical_batch": (
        "utils.training.ldf.batch",
        "anchor_physical_batch",
    ),
    "build_ldf_training_step": (
        "utils.training.ldf.batch",
        "build_ldf_training_step",
    ),
    "create_xz_condition": (
        "utils.training.ldf.conditioning",
        "create_xz_condition",
    ),
    "compute_velocity_loss": (
        "utils.training.ldf.losses",
        "compute_velocity_loss",
    ),
    "compute_offpath_loss": (
        "utils.training.ldf.losses",
        "compute_offpath_loss",
    ),
    "create_dataloaders": ("utils.training.ldf.data", "create_dataloaders"),
    "create_dataset": ("utils.training.ldf.data", "create_dataset"),
    "resolve_self_forcing_k": (
        "utils.training.ldf.self_forcing",
        "resolve_self_forcing_k",
    ),
    "run_self_forcing_rollout": (
        "utils.training.ldf.self_forcing",
        "run_self_forcing_rollout",
    ),
    "sample_rollout_steps": (
        "utils.training.ldf.self_forcing",
        "sample_rollout_steps",
    ),
    "sample_constraint_keep_mask": (
        "utils.training.ldf.conditioning",
        "sample_constraint_keep_mask",
    ),
    "sample_xz_constraint_mask": (
        "utils.training.ldf.conditioning",
        "sample_xz_constraint_mask",
    ),
    "sample_window_plan": (
        "utils.training.ldf.self_forcing",
        "sample_window_plan",
    ),
    "self_forcing_phase_progress": (
        "utils.training.ldf.self_forcing",
        "self_forcing_phase_progress",
    ),
    "validate_self_forcing_config": (
        "utils.training.ldf.self_forcing",
        "validate_self_forcing_config",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
