"""Shared conversion primitives for HumanML-style 263D motion artifacts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from utils.conditions.vae import CONTRACT_VERSION, FRAMES_PER_TOKEN
from utils.motion_representation import (
    HUMANML_DIM,
    HUMANML_SOURCE_REPRESENTATION,
    humanml263_to_root_body_motion,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def artifact_is_current(source: Path, target: Path) -> bool:
    if not target.is_file():
        return False
    try:
        with np.load(target, allow_pickle=False) as data:
            return (
                str(np.asarray(data["contract_version"]).item()) == CONTRACT_VERSION
                and str(np.asarray(data["source_representation"]).item())
                == HUMANML_SOURCE_REPRESENTATION
                and str(np.asarray(data["source_sha256"]).item())
                == sha256_file(source)
            )
    except (KeyError, OSError, ValueError):
        return False


def process_file(source: Path, target: Path, *, fps: float = 20.0) -> dict:
    feature = np.load(source, allow_pickle=False)
    if feature.ndim != 2 or feature.shape[-1] != HUMANML_DIM:
        raise ValueError(
            f"HumanML-style source must be [F,{HUMANML_DIM}], "
            f"got {feature.shape} at {source}"
        )
    if not np.isfinite(feature).all():
        raise ValueError(f"HumanML-style source contains non-finite values at {source}")
    usable = feature.shape[0] // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN
    if usable < FRAMES_PER_TOKEN:
        raise ValueError(f"{source} has fewer than four usable frames")
    motion = torch.from_numpy(feature[:usable]).float()
    root, body, feature_valid = humanml263_to_root_body_motion(motion, fps=fps)
    source_sha256 = sha256_file(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            root_motion=root.numpy(),
            body_motion=body.numpy(),
            body_feature_valid_mask=feature_valid.numpy(),
            contract_version=CONTRACT_VERSION,
            source_representation=HUMANML_SOURCE_REPRESENTATION,
            fps=np.float32(fps),
            source_sha256=source_sha256,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": usable,
        "source_representation": HUMANML_SOURCE_REPRESENTATION,
        "source_sha256": source_sha256,
    }


__all__ = [
    "artifact_is_current",
    "atomic_write_text",
    "process_file",
    "sha256_file",
]
