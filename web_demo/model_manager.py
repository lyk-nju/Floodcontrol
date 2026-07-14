"""Explicit migration guard for the temporarily unavailable web generator."""

from __future__ import annotations


WEB_MIGRATION_ERROR = (
    "Floodcontrol Web generation is BLOCKED_ON_STRICT4_VAE: the external planner "
    "and legacy full-motion LDF were removed. Connect the strict four-frame body "
    "VAE, the new data protocol, and commit-time decoder transaction before "
    "starting the Web runtime."
)


class ModelManager:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(WEB_MIGRATION_ERROR)


def get_model_manager(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(WEB_MIGRATION_ERROR)


__all__ = ["ModelManager", "WEB_MIGRATION_ERROR", "get_model_manager"]
