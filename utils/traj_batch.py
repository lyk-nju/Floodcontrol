"""Legacy dataset geometry helpers retained until the data-pipeline rewrite.

This module no longer creates learned trajectory embeddings.  The remaining
functions are pure preprocessing utilities still used by the copied datasets.
"""

from __future__ import annotations

import numpy as np
import torch


_PATH_HEADING_EPS = 1e-8


def smooth_root_xz(root_xz: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth root XZ positions; ``sigma <= 0`` returns a copy."""
    if sigma <= 0.0:
        return root_xz.astype(np.float32)
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(
        root_xz.astype(np.float64), sigma=sigma, axis=0
    ).astype(np.float32)


def root_to_traj_feats(traj_xyz, eps: float = _PATH_HEADING_EPS):
    """Convert root XYZ to the old pure `[x,z,cos,sin]` dataset view.

    This representation is not consumed by the new LDF.  It remains only so
    old datasets can be inspected before their strict hybrid rewrite.
    """
    if isinstance(traj_xyz, np.ndarray):
        arr = np.asarray(traj_xyz, dtype=np.float64)
        x, z = arr[:, 0:1], arr[:, 2:3]
        if arr.shape[0] == 1:
            return np.concatenate(
                [x, z, np.ones_like(x), np.zeros_like(z)], axis=-1
            ).astype(np.float32)
        dx = np.empty_like(x)
        dz = np.empty_like(z)
        dx[:1], dz[:1] = x[1:2] - x[:1], z[1:2] - z[:1]
        dx[1:], dz[1:] = x[1:] - x[:-1], z[1:] - z[:-1]
        norm = np.sqrt(np.maximum(dx * dx + dz * dz, eps * eps))
        short = (dx * dx + dz * dz) < eps * eps
        return np.concatenate(
            [
                x,
                z,
                np.where(short, 1.0, dx / norm),
                np.where(short, 0.0, dz / norm),
            ],
            axis=-1,
        ).astype(np.float32)

    x, z = traj_xyz[..., 0:1], traj_xyz[..., 2:3]
    if x.shape[-2] == 1:
        return torch.cat([x, z, torch.ones_like(x), torch.zeros_like(z)], dim=-1)
    dx, dz = torch.empty_like(x), torch.empty_like(z)
    dx[..., :1, :], dz[..., :1, :] = x[..., 1:2, :] - x[..., :1, :], z[
        ..., 1:2, :
    ] - z[..., :1, :]
    dx[..., 1:, :], dz[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :], z[
        ..., 1:, :
    ] - z[..., :-1, :]
    squared = dx * dx + dz * dz
    norm = squared.sqrt().clamp_min(eps)
    short = squared < eps * eps
    return torch.cat(
        [
            x,
            z,
            torch.where(short, torch.ones_like(dx), dx / norm),
            torch.where(short, torch.zeros_like(dz), dz / norm),
        ],
        dim=-1,
    )


__all__ = ["root_to_traj_feats", "smooth_root_xz"]
