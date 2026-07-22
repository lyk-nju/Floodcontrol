"""Build minimal root5/body259 NPZ files from HumanML-style 263D motion."""

from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import torch

from tools.convert_motion_263_to_259 import (
    HUMANML_DIM,
    convert_motion_263_to_259,
)
from utils.motion_process import BODY_DIM, ROOT_DIM
from utils.token_frame import FRAMES_PER_TOKEN, aligned_frame_floor


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_is_current(source: Path, target: Path, *, fps: float) -> bool:
    """Return whether a usable minimal target already exists.

    ``source`` and ``fps`` remain in the signature so preprocessing worker code
    stays simple.  Dataset products intentionally carry no source hashes or
    converter/runtime version identity.
    """

    del source, fps
    if not target.is_file():
        return False
    try:
        with np.load(target, allow_pickle=False) as data:
            if set(data.files) != {
                "root_motion",
                "body_motion",
                "body_feature_valid_mask",
            }:
                return False
            root = np.asarray(data["root_motion"])
            body = np.asarray(data["body_motion"])
            valid = np.asarray(data["body_feature_valid_mask"])
            heading_norm = np.linalg.norm(root[:, 3:5], axis=-1)
            return bool(
                root.ndim == 2
                and root.shape[-1] == ROOT_DIM
                and root.dtype == np.float32
                and body.ndim == 2
                and body.shape[-1] == BODY_DIM
                and body.dtype == np.float32
                and valid.shape == body.shape
                and valid.dtype == np.bool_
                and root.shape[0] == body.shape[0]
                and root.shape[0] >= FRAMES_PER_TOKEN
                and root.shape[0] % FRAMES_PER_TOKEN == 0
                and np.isfinite(root).all()
                and np.isfinite(body).all()
                and np.allclose(heading_norm, 1.0, atol=1e-5, rtol=0.0)
            )
    except (IndexError, KeyError, OSError, ValueError):
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
    usable = aligned_frame_floor(feature.shape[0])
    if usable < FRAMES_PER_TOKEN:
        raise ValueError(f"{source} has fewer than four usable frames")
    motion = torch.from_numpy(feature[:usable]).float()
    root, body, feature_valid = convert_motion_263_to_259(motion, fps=fps)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            root_motion=root.numpy(),
            body_motion=body.numpy(),
            body_feature_valid_mask=feature_valid.numpy(),
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"frames": usable}


__all__ = [
    "artifact_is_current",
    "atomic_copy",
    "atomic_write_text",
    "process_file",
]
