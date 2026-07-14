"""BABEL root5/body265 artifact Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .humanml3d import (
    HumanML3DDataset,
    _load_records,
    collate_humanml3d,
)


def load_babel_records(
    meta_paths: Iterable[str | Path],
    *,
    artifact_path: str = "artifacts",
) -> list[dict[str, object]]:
    return _load_records(
        meta_paths,
        artifact_path=artifact_path,
        dataset_label="BABEL",
    )


class BABELDataset(HumanML3DDataset):
    """Consume versioned BABEL_motion artifacts using the shared VAE contract."""

    @staticmethod
    def load_records(
        meta_paths: Iterable[str | Path],
        *,
        artifact_path: str,
    ) -> list[dict[str, object]]:
        return load_babel_records(meta_paths, artifact_path=artifact_path)


def collate_babel(batch: list[dict]) -> dict[str, object]:
    return collate_humanml3d(batch)


__all__ = ["BABELDataset", "collate_babel", "load_babel_records"]
