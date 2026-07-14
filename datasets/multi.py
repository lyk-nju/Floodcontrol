"""Composition of homogeneous root5/body265 motion Datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from torch.utils.data import ConcatDataset

from .humanml3d import collate_humanml3d
from utils.initialize import instantiate


class MultiDataset(ConcatDataset):
    """Concatenate source-specific Datasets that share the VAE batch contract."""

    def __init__(
        self,
        *,
        dataset_configs: Iterable[Mapping[str, object]],
        split: str,
        min_frames: int = 20,
        max_frames: int = 200,
        random_yaw: bool = False,
        expected_fps: float = 20.0,
    ):
        datasets = []
        for index, dataset_config in enumerate(dataset_configs):
            target = dataset_config.get("target")
            if not target:
                raise ValueError(f"multi dataset entry {index} is missing target")
            meta_paths = dataset_config.get(f"{split}_meta_paths")
            if not meta_paths:
                raise ValueError(
                    f"multi dataset entry {index} is missing {split}_meta_paths"
                )
            datasets.append(
                instantiate(
                    str(target),
                    cfg=None,
                    meta_paths=meta_paths,
                    split=split,
                    artifact_path=dataset_config.get("artifact_path", "artifacts"),
                    min_frames=dataset_config.get("min_frames", min_frames),
                    max_frames=dataset_config.get("max_frames", max_frames),
                    random_yaw=dataset_config.get("random_yaw", random_yaw),
                    expected_fps=dataset_config.get("expected_fps", expected_fps),
                )
            )
        if not datasets:
            raise ValueError("MultiDataset requires at least one dataset entry")
        super().__init__(datasets)

    @property
    def dataset_lengths(self) -> tuple[int, ...]:
        starts = (0, *self.cumulative_sizes[:-1])
        return tuple(end - start for start, end in zip(starts, self.cumulative_sizes))


def collate_multi(batch: list[dict]) -> dict[str, object]:
    """Collate the one shared VAE contract; source-specific keys are forbidden."""
    return collate_humanml3d(batch)


__all__ = ["MultiDataset", "collate_multi"]
