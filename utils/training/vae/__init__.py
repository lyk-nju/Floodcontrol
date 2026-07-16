"""Lazy public exports for body VAE training components."""

from importlib import import_module

__all__ = [
    "VAELightningModule",
    "VAELoss",
    "VAEWindowCollator",
    "create_dataloaders",
    "create_dataset",
]


_EXPORTS = {
    "VAELightningModule": (
        "utils.training.vae.lightning_module",
        "VAELightningModule",
    ),
    "VAELoss": ("utils.training.vae.losses", "VAELoss"),
    "VAEWindowCollator": ("utils.training.vae.data", "VAEWindowCollator"),
    "create_dataloaders": ("utils.training.vae.data", "create_dataloaders"),
    "create_dataset": ("utils.training.vae.data", "create_dataset"),
}


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
