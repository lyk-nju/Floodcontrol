from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from models.vae_wan_1d import BodyVAE


def write_statistics(
    directory: str | Path,
    *,
    latent_dim: int | None = None,
    latent_mean: float = 0.0,
    latent_std: float = 1.0,
) -> tuple[Path, Path | None]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    motion = directory / "motion_stats.npz"
    np.savez(
        motion,
        body_cont_mean=np.zeros(261, dtype=np.float32),
        body_cont_std=np.ones(261, dtype=np.float32),
        local_root_mean=np.zeros(4, dtype=np.float32),
        local_root_std=np.ones(4, dtype=np.float32),
    )
    latent = None
    if latent_dim is not None:
        latent = directory / "latent_stats.npz"
        np.savez(
            latent,
            mean=np.full(latent_dim, latent_mean, dtype=np.float32),
            std=np.full(latent_dim, latent_std, dtype=np.float32),
        )
    return motion, latent


def make_vae(*, with_latent_stats: bool = True, **kwargs) -> BodyVAE:
    latent_dim = int(kwargs.get("latent_dim", 128))
    with TemporaryDirectory() as directory:
        motion, latent = write_statistics(
            directory, latent_dim=latent_dim if with_latent_stats else None
        )
        return BodyVAE(
            motion_stats_path=motion,
            latent_stats_path=latent,
            **kwargs,
        )
