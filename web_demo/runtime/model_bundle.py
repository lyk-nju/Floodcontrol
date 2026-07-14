"""Loaded model bundle contract for the web runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    vae: Any
    ldf_model: Any
    cfg: Any
    device: str
    stream_kernel: Any | None = None
    runtime_session: Any | None = None


__all__ = ["ModelBundle"]
