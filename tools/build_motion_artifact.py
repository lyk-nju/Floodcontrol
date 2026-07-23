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
from utils.motion_process import (
    BODY_DIM,
    CONTACT_SLICE,
    ROOT_DIM,
    VELOCITY_SLICE,
)
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


def artifact_arrays_are_current(
    root: np.ndarray,
    body: np.ndarray,
    valid: np.ndarray,
) -> bool:
    """Validate the complete minimal artifact contract without metadata.

    The cold-start validity pattern is intentionally part of the numeric
    contract.  Besides rejecting malformed arrays, it distinguishes the
    backward/current contact representation from older Body259 artifacts that
    copied HumanML's forward contact labels unchanged.
    """

    if (
        root.ndim != 2
        or root.shape[-1] != ROOT_DIM
        or root.dtype != np.float32
        or body.ndim != 2
        or body.shape[-1] != BODY_DIM
        or body.dtype != np.float32
        or valid.shape != body.shape
        or valid.dtype != np.bool_
        or root.shape[0] != body.shape[0]
        or root.shape[0] < FRAMES_PER_TOKEN
        or root.shape[0] % FRAMES_PER_TOKEN != 0
        or not np.isfinite(root).all()
        or not np.isfinite(body).all()
    ):
        return False
    heading_norm = np.linalg.norm(root[:, 3:5], axis=-1)
    if not np.allclose(heading_norm, 1.0, atol=1e-5, rtol=0.0):
        return False
    if not valid[:, : VELOCITY_SLICE.start].all():
        return False
    if valid[0, VELOCITY_SLICE].any() or valid[0, CONTACT_SLICE].any():
        return False
    if not valid[1:, VELOCITY_SLICE].all() or not valid[1:, CONTACT_SLICE].all():
        return False
    return bool(
        np.allclose(body[0, VELOCITY_SLICE], 0.0)
        and np.allclose(body[0, CONTACT_SLICE], 0.0)
    )


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
            return artifact_arrays_are_current(root, body, valid)
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
    "artifact_arrays_are_current",
    "artifact_is_current",
    "atomic_copy",
    "atomic_write_text",
    "process_file",
]
