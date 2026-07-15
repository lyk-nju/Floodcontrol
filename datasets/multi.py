"""Composition of source-specific Datasets sharing one full-motion contract."""

from __future__ import annotations
from collections.abc import Iterable, Mapping
from torch.utils.data import ConcatDataset, Dataset
from utils.initialize import instantiate_target


class MultiDataset(ConcatDataset):
    """Concatenate source Datasets that return the same complete sample contract."""

    def __init__(
        self,
        *,
        dataset_configs: Iterable[Mapping[str, object]],
        split: str,
        fps: float = 20.0,
    ):
        self.split = str(split)
        datasets: list[Dataset] = []

        for dataset_config in dataset_configs:
            # Pass only this source's paths and options.  Source Datasets do not
            # see or inherit settings belonging to the other entries.
            dataset = instantiate_target(
                str(dataset_config["target"]),
                cfg=None,
                meta_paths=dataset_config[f"{self.split}_meta_paths"],
                split=self.split,
                artifact_path=dataset_config.get("artifact_path", "artifacts"),
                text_path=dataset_config.get("text_path"),
                fps=dataset_config.get("fps", fps),
            )
            datasets.append(dataset)

        if not datasets:
            raise ValueError("MultiDataset requires at least one dataset entry")
        super().__init__(datasets)

    @property
    def dataset_lengths(self) -> tuple[int, ...]:
        return tuple(len(dataset) for dataset in self.datasets)


__all__ = ["MultiDataset"]
