"""Shared model condition contracts."""

from .ldf import (
    HybridMotion,
    LDFCondition,
    LDFInput,
    LDFPrediction,
    LDFStreamState,
    create_cfg_condition,
    create_ldf_condition,
    create_window_condition,
    derive_local_root_motion,
)
from .vae import (
    BODY_CONTINUOUS_DIM,
    BODY_DIM,
    CONTRACT_VERSION as VAE_CONTRACT_VERSION,
    BodyPrediction,
    VAEDecoderState,
    VAEInput,
    VAEPosterior,
    VAEPrediction,
)

__all__ = [
    "HybridMotion",
    "LDFCondition",
    "LDFInput",
    "LDFPrediction",
    "LDFStreamState",
    "create_cfg_condition",
    "create_ldf_condition",
    "create_window_condition",
    "derive_local_root_motion",
    "BODY_CONTINUOUS_DIM",
    "BODY_DIM",
    "VAE_CONTRACT_VERSION",
    "BodyPrediction",
    "VAEDecoderState",
    "VAEInput",
    "VAEPosterior",
    "VAEPrediction",
]
