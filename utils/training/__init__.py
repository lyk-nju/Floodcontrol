"""Lazy public exports for training utilities."""

from importlib import import_module

__all__ = [
    "BasicLightningModule",
    "LDFLightningModule",
    "VAELightningModule",
    "VAELoss",
]


_EXPORTS = {
    "BasicLightningModule": (
        "utils.training.lightning_module",
        "BasicLightningModule",
    ),
    "LDFLightningModule": (
        "utils.training.ldf.lightning_module",
        "LDFLightningModule",
    ),
    "VAELightningModule": (
        "utils.training.vae.lightning_module",
        "VAELightningModule",
    ),
    "VAELoss": ("utils.training.vae.losses", "VAELoss"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
