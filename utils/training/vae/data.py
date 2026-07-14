"""Dataset and dataloader construction for body VAE training."""

from __future__ import annotations

from torch.utils.data import DataLoader

from utils.initialize import get_function, instantiate


def create_dataset(cfg, split: str):
    common_args = {
        "split": split,
        "min_frames": cfg.data.min_frames,
        "max_frames": cfg.data.max_frames,
        "random_yaw": cfg.data.random_yaw,
        "expected_fps": cfg.model.params.fps,
    }
    dataset_configs = cfg.data.get("datasets", None)
    if dataset_configs:
        return instantiate(
            cfg.data.target,
            cfg=None,
            dataset_configs=dataset_configs,
            **common_args,
        )
    meta_paths = cfg.data.get(f"{split}_meta_paths", None)
    if not meta_paths:
        raise RuntimeError(
            f"MOTION_ARTIFACT_DATA_REQUIRED: set data.{split}_meta_paths to "
            "sample-id TXT files backed by preprocessed HumanML3D root5/body265 "
            "artifacts. The training Dataset does not convert 263D motions online."
        )
    return instantiate(
        cfg.data.target,
        cfg=None,
        meta_paths=meta_paths,
        artifact_path=cfg.data.artifact_path,
        **common_args,
    )


def create_dataloaders(cfg) -> tuple[DataLoader | None, DataLoader]:
    train_dataset = create_dataset(cfg, "train") if cfg.train else None
    val_dataset = create_dataset(cfg, "val")
    collate_fn = get_function(cfg.data.collate_fn)
    loader_kwargs = {
        "num_workers": cfg.data.num_workers,
        "collate_fn": collate_fn,
    }
    train_dataloader = (
        DataLoader(
            train_dataset,
            batch_size=cfg.data.train_bs,
            shuffle=True,
            **loader_kwargs,
        )
        if train_dataset is not None
        else None
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_bs,
        shuffle=False,
        **loader_kwargs,
    )
    return train_dataloader, val_dataloader


__all__ = ["create_dataloaders", "create_dataset"]
