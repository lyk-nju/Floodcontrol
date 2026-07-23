"""Dataset construction and task-specific window collation for BodyVAE training."""

from __future__ import annotations

import random

import torch
from torch.utils.data import DataLoader

from utils.initialize import instantiate_target
from utils.motion_process import (
    BODY_DIM,
    ROOT_DIM,
    rotate_motion_yaw,
    rotate_root_yaw,
)
from utils.token_frame import FRAMES_PER_TOKEN, MOTION_FPS


class VAEWindowCollator:
    """Crop full physical clips and construct the padded BodyVAE batch contract."""

    def __init__(
        self,
        *,
        min_frames: int,
        max_frames: int,
        training: bool,
        random_yaw: bool = False,
    ):
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.training = bool(training)
        self.random_yaw = bool(random_yaw and training)
        for name, value in (
            ("min_frames", self.min_frames),
            ("max_frames", self.max_frames),
        ):
            if value < FRAMES_PER_TOKEN or value % FRAMES_PER_TOKEN:
                raise ValueError(f"{name} must be a positive multiple of four")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must not exceed max_frames")

    def _window(self, sample: dict[str, object]) -> dict[str, object]:
        full_root = sample["root_motion"]
        full_body = sample["body_motion"]
        full_valid = sample["body_feature_valid_mask"]
        available = int(full_root.shape[0])
        if available < self.min_frames:
            raise ValueError(
                f"sample {sample['dataset']}/{sample['name']} has {available} frames, "
                f"fewer than the required {self.min_frames}"
            )
        maximum = min(available, self.max_frames)
        if self.training:
            token_count = random.randint(
                self.min_frames // FRAMES_PER_TOKEN,
                maximum // FRAMES_PER_TOKEN,
            )
            frames = token_count * FRAMES_PER_TOKEN
            start_token = random.randint(
                0, (available - frames) // FRAMES_PER_TOKEN
            )
            start = start_token * FRAMES_PER_TOKEN
        else:
            frames = maximum // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN
            start = 0

        previous = full_root[start - 1].clone() if start > 0 else None
        root = full_root[start : start + frames].clone()
        body = full_body[start : start + frames].clone()
        feature_valid = full_valid[start : start + frames].clone()

        origin_x, origin_z = root[0, 0].clone(), root[0, 2].clone()
        root[:, 0] -= origin_x
        root[:, 2] -= origin_z
        if previous is not None:
            previous[0] -= origin_x
            previous[2] -= origin_z

        if self.random_yaw:
            angle = torch.rand(1) * (2.0 * torch.pi)
            root, body = rotate_motion_yaw(root[None], body[None], angle)
            root, body = root[0], body[0]
            if previous is not None:
                previous = rotate_root_yaw(previous[None, None], angle)[0, 0]

        return {
            "dataset": sample["dataset"],
            "name": sample["name"],
            "root_motion": root,
            "body_motion": body,
            "body_feature_valid_mask": feature_valid,
            "previous_root_frame": previous,
            "text_data": sample.get("text_data", []),
        }

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        if not samples:
            raise ValueError("VAEWindowCollator requires a non-empty batch")
        windows = [self._window(sample) for sample in samples]
        max_frames = max(int(item["body_motion"].shape[0]) for item in windows)
        batch_size = len(windows)
        body = torch.zeros(batch_size, max_frames, BODY_DIM)
        root = torch.zeros(batch_size, max_frames, ROOT_DIM)
        # Padding remains mask-invalid, but every Root5 value must still be a
        # structurally valid physical state.  A zero heading cannot be passed
        # through world-position/FK diagnostics before masking, so use the
        # neutral unit heading for padded frames.
        root[..., 3] = 1.0
        frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
        feature_mask = torch.zeros(batch_size, max_frames, BODY_DIM, dtype=torch.bool)
        previous = torch.zeros(batch_size, ROOT_DIM)
        previous[..., 3] = 1.0
        previous_valid = torch.zeros(batch_size, dtype=torch.bool)
        for index, item in enumerate(windows):
            frames = int(item["body_motion"].shape[0])
            body[index, :frames] = item["body_motion"]
            root[index, :frames] = item["root_motion"]
            frame_mask[index, :frames] = True
            feature_mask[index, :frames] = item["body_feature_valid_mask"]
            if item["previous_root_frame"] is not None:
                previous[index] = item["previous_root_frame"]
                previous_valid[index] = True
        return {
            "body_motion": body,
            "root_motion": root,
            "frame_valid_mask": frame_mask,
            "body_feature_valid_mask": feature_mask,
            "previous_root_frame": previous,
            "previous_root_valid_mask": previous_valid,
            "dataset": [str(item["dataset"]) for item in windows],
            "name": [str(item["name"]) for item in windows],
            "text_data": [item["text_data"] for item in windows],
        }


def create_dataset(cfg, split: str):
    common_args = {"split": split, "fps": MOTION_FPS}
    dataset_configs = cfg.data.get("datasets", None)
    if dataset_configs:
        return instantiate_target(
            cfg.data.target,
            cfg=None,
            dataset_configs=dataset_configs,
            **common_args,
        )
    meta_paths = cfg.data.get(f"{split}_meta_paths", None)
    if not meta_paths:
        raise RuntimeError(f"set data.{split}_meta_paths to processed motion split files")
    return instantiate_target(
        cfg.data.target,
        cfg=None,
        meta_paths=meta_paths,
        artifact_path=cfg.data.get("artifact_path", "artifacts"),
        text_path=cfg.data.get("text_path"),
        **common_args,
    )


def create_dataloaders(cfg) -> tuple[DataLoader | None, DataLoader]:
    train_dataset = create_dataset(cfg, "train") if cfg.train else None
    val_dataset = create_dataset(cfg, "val")
    common = {"num_workers": int(cfg.data.num_workers)}
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=int(cfg.data.train_batch_size),
            shuffle=True,
            collate_fn=VAEWindowCollator(
                min_frames=cfg.data.min_frames,
                max_frames=cfg.data.max_frames,
                training=True,
                random_yaw=cfg.data.random_yaw,
            ),
            **common,
        )
        if train_dataset is not None
        else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.data.val_batch_size),
        shuffle=False,
        collate_fn=VAEWindowCollator(
            min_frames=cfg.data.min_frames,
            max_frames=cfg.data.max_frames,
            training=False,
            random_yaw=False,
        ),
        **common,
    )
    return train_loader, val_loader


__all__ = [
    "VAEWindowCollator",
    "create_dataloaders",
    "create_dataset",
]
