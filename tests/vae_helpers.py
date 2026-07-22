from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from models.vae_wan_1d import BodyVAE


def write_statistics(
    directory: str | Path,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    motion = directory / "motion_stats.npz"
    np.savez(
        motion,
        body_cont_mean=np.zeros(255, dtype=np.float32),
        body_cont_std=np.ones(255, dtype=np.float32),
        local_root_mean=np.zeros(4, dtype=np.float32),
        local_root_std=np.ones(4, dtype=np.float32),
    )
    return motion


def make_vae(**kwargs) -> BodyVAE:
    with TemporaryDirectory() as directory:
        motion = write_statistics(directory)
        return BodyVAE(
            motion_stats_path=motion,
            **kwargs,
        )
