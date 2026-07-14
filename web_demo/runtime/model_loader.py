"""Fail-fast model loader pending strict-4 VAE/runtime integration."""

from __future__ import annotations

import os

from web_demo.model_manager import WEB_MIGRATION_ERROR


def resolve_repo_path(path):
    if not path:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.abspath(os.path.join(root, path))


def _blocked(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(WEB_MIGRATION_ERROR)


load_ldf_models = _blocked
load_model_bundle = _blocked
build_runtime_session = _blocked


__all__ = [
    "build_runtime_session",
    "load_ldf_models",
    "load_model_bundle",
    "resolve_repo_path",
]
