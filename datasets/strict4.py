"""Dataset for versioned root5/body265 strict-4 artifacts."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.conditions.vae import BODY_DIM, CONTRACT_VERSION, FRAMES_PER_TOKEN, ROOT_DIM
from utils.motion_representation import rotate_root_body_yaw


class Strict4ArtifactDataset(Dataset):
    def __init__(
        self,
        *,
        manifest_path: str,
        split: str,
        min_frames: int = 20,
        max_frames: int = 200,
        random_yaw: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise RuntimeError(
                "STRICT4_NATIVE_ROTATIONS_REQUIRED: manifest not found at "
                f"{self.manifest_path}. Build it from native SMPL/AMASS rotations; "
                "legacy 263D and IK fallbacks are intentionally unsupported."
            )
        self.root = self.manifest_path.parent
        self.split = str(split)
        self.min_frames = max(FRAMES_PER_TOKEN, int(min_frames) // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN)
        self.max_frames = int(max_frames) // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN
        self.random_yaw = bool(random_yaw and split == "train")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must not exceed max_frames")
        self.records = []
        for line in self.manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("contract_version") != CONTRACT_VERSION:
                raise ValueError("strict4 manifest contract version mismatch")
            if record.get("split") == self.split:
                self.records.append(record)
        if not self.records:
            raise RuntimeError(f"strict4 manifest contains no {self.split!r} samples")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        path = self.root / record["artifact"]
        with np.load(path, allow_pickle=False) as data:
            root = torch.from_numpy(data["root_motion"]).float()
            body = torch.from_numpy(data["body_motion"]).float()
            feature_valid = torch.from_numpy(data["body_feature_valid_mask"]).bool()
            previous_root = (
                torch.from_numpy(data["previous_root_frame"]).float()
                if "previous_root_frame" in data else None
            )
        if root.ndim != 2 or root.shape[-1] != ROOT_DIM or body.ndim != 2 or body.shape[-1] != BODY_DIM:
            raise ValueError(f"invalid strict4 artifact shapes in {path}")
        available = min(root.shape[0], body.shape[0]) // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN
        if available < self.min_frames:
            raise ValueError(f"artifact {path} is shorter than {self.min_frames} frames")
        crop_length = min(available, self.max_frames)
        if self.split == "train" and crop_length > self.min_frames:
            token_count = random.randint(
                self.min_frames // FRAMES_PER_TOKEN,
                crop_length // FRAMES_PER_TOKEN,
            )
            crop_length = token_count * FRAMES_PER_TOKEN
        max_start_token = (available - crop_length) // FRAMES_PER_TOKEN
        start = (random.randint(0, max_start_token) if self.split == "train" else 0) * FRAMES_PER_TOKEN
        if start > 0:
            previous_root = root[start - 1].clone()
        root = root[start : start + crop_length]
        body = body[start : start + crop_length]
        feature_valid = feature_valid[start : start + crop_length]
        root = root.clone()
        origin_x = root[0, 0].clone()
        origin_z = root[0, 2].clone()
        root[:, 0] -= origin_x
        root[:, 2] -= origin_z
        if previous_root is not None:
            previous_root = previous_root.clone()
            previous_root[0] -= origin_x
            previous_root[2] -= origin_z
        if self.random_yaw:
            angle = torch.rand(1) * (2 * torch.pi)
            root, body = rotate_root_body_yaw(root[None], body[None], angle)
            root, body = root[0], body[0]
            if previous_root is not None:
                previous_root, _ = rotate_root_body_yaw(
                    previous_root[None, None], body[:1][None], angle
                )
                previous_root = previous_root[0, 0]
        return {
            "body_motion": body,
            "root_motion": root,
            "body_feature_valid_mask": feature_valid,
            "previous_root_frame": previous_root,
            "name": record.get("name", path.stem),
            "text": record.get("text", ""),
        }


def collate_strict4(batch: list[dict]) -> dict[str, object]:
    max_frames = max(item["body_motion"].shape[0] for item in batch)
    if max_frames % FRAMES_PER_TOKEN:
        raise AssertionError("collate padding must be token aligned")
    batch_size = len(batch)
    body = torch.zeros(batch_size, max_frames, BODY_DIM)
    root = torch.zeros(batch_size, max_frames, ROOT_DIM)
    frame_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    feature_mask = torch.zeros(batch_size, max_frames, BODY_DIM, dtype=torch.bool)
    previous = torch.zeros(batch_size, ROOT_DIM)
    previous_valid = torch.zeros(batch_size, dtype=torch.bool)
    for idx, item in enumerate(batch):
        frames = item["body_motion"].shape[0]
        body[idx, :frames] = item["body_motion"]
        root[idx, :frames] = item["root_motion"]
        frame_mask[idx, :frames] = True
        feature_mask[idx, :frames] = item["body_feature_valid_mask"]
        if item["previous_root_frame"] is None:
            pass
        else:
            previous[idx] = item["previous_root_frame"]
            previous_valid[idx] = True
    return {
        "body_motion": body,
        "root_motion": root,
        "frame_valid_mask": frame_mask,
        "body_feature_valid_mask": feature_mask,
        "previous_root_frame": previous,
        "previous_root_valid_mask": previous_valid,
        "name": [item["name"] for item in batch],
        "text": [item["text"] for item in batch],
    }


__all__ = ["Strict4ArtifactDataset", "collate_strict4"]
