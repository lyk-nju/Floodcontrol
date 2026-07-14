"""HumanML3D root5/body265 artifact Dataset."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.conditions.vae import (
    BODY_DIM,
    CONTRACT_VERSION,
    FRAMES_PER_TOKEN,
    ROOT_DIM,
)
from utils.motion_representation import (
    HUMANML_SOURCE_REPRESENTATION,
    rotate_root_body_yaw,
    rotate_root_yaw,
)


SUPPORTED_SPLITS = frozenset({"train", "val", "test"})


def _read_scalar(data, name: str, path: Path):
    if name not in data:
        raise ValueError(f"motion artifact is missing {name!r} in {path}")
    value = np.asarray(data[name])
    if value.shape != ():
        raise ValueError(f"motion artifact {name!r} must be scalar in {path}")
    return value.item()


def _load_records(
    meta_paths: Iterable[str | Path],
    *,
    artifact_path: str = "artifacts",
    dataset_label: str,
) -> list[dict[str, object]]:
    """Resolve sample-id TXT files to one processed dataset's artifacts."""
    artifact_subdir = Path(artifact_path)
    if artifact_subdir.is_absolute() or ".." in artifact_subdir.parts:
        raise ValueError("artifact_path must be a relative directory")
    records: list[dict[str, object]] = []
    for meta_value in meta_paths:
        meta_path = Path(meta_value)
        if not meta_path.is_file():
            raise RuntimeError(
                "MOTION_ARTIFACT_DATA_REQUIRED: split metadata file not found at "
                f"{meta_path}. Preprocess {dataset_label} into root5/body265 artifacts."
            )
        dataset_name = meta_path.parent.name
        for line_number, line in enumerate(meta_path.read_text().splitlines(), start=1):
            name = line.strip()
            if not name:
                continue
            if Path(name).name != name:
                raise ValueError(
                    f"sample id must be a plain filename stem: {meta_path}:{line_number}"
                )
            artifact = meta_path.parent / artifact_subdir / f"{name}.npz"
            if not artifact.is_file():
                raise RuntimeError(
                    "MOTION_ARTIFACT_DATA_REQUIRED: artifact missing for sample "
                    f"{name!r} at {artifact}"
                )
            records.append(
                {"name": name, "dataset": dataset_name, "artifact": artifact}
            )
    if not records:
        raise RuntimeError("motion split metadata contains no samples")
    return records


def load_humanml3d_records(
    meta_paths: Iterable[str | Path],
    *,
    artifact_path: str = "artifacts",
) -> list[dict[str, object]]:
    return _load_records(
        meta_paths,
        artifact_path=artifact_path,
        dataset_label="HumanML3D",
    )


class HumanML3DDataset(Dataset):
    SOURCE_REPRESENTATION = HUMANML_SOURCE_REPRESENTATION

    @staticmethod
    def load_records(
        meta_paths: Iterable[str | Path],
        *,
        artifact_path: str,
    ) -> list[dict[str, object]]:
        return load_humanml3d_records(meta_paths, artifact_path=artifact_path)

    def __init__(
        self,
        *,
        meta_paths: Iterable[str | Path],
        split: str,
        artifact_path: str = "artifacts",
        min_frames: int = 20,
        max_frames: int = 200,
        random_yaw: bool = False,
        expected_fps: float = 20.0,
    ):
        self.split = str(split)
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"unsupported split {self.split!r}; expected one of {sorted(SUPPORTED_SPLITS)}"
            )
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        for name, original, value in (
            ("min_frames", min_frames, self.min_frames),
            ("max_frames", max_frames, self.max_frames),
        ):
            if (
                value != original
                or value < FRAMES_PER_TOKEN
                or value % FRAMES_PER_TOKEN
            ):
                raise ValueError(f"{name} must be a positive multiple of four")
        self.expected_fps = float(expected_fps)
        if not math.isfinite(self.expected_fps) or self.expected_fps <= 0:
            raise ValueError("expected_fps must be finite and positive")
        self.random_yaw = bool(random_yaw and self.split == "train")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must not exceed max_frames")
        self.records = self.load_records(
            meta_paths, artifact_path=artifact_path
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        path = Path(record["artifact"])
        with np.load(path, allow_pickle=False) as data:
            if str(_read_scalar(data, "contract_version", path)) != CONTRACT_VERSION:
                raise ValueError(f"motion artifact contract version mismatch in {path}")
            source_representation = str(
                _read_scalar(data, "source_representation", path)
            )
            if source_representation != self.SOURCE_REPRESENTATION:
                raise ValueError(
                    "motion artifact source representation mismatch in "
                    f"{path}: expected {self.SOURCE_REPRESENTATION!r}, "
                    f"got {source_representation!r}"
                )
            artifact_fps = float(_read_scalar(data, "fps", path))
            if not math.isclose(
                artifact_fps, self.expected_fps, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"motion artifact FPS mismatch in {path}: "
                    f"expected {self.expected_fps}, got {artifact_fps}"
                )
            for name in (
                "root_motion",
                "body_motion",
                "body_feature_valid_mask",
            ):
                if name not in data:
                    raise ValueError(f"motion artifact is missing {name!r} in {path}")
            root = torch.from_numpy(data["root_motion"]).float()
            body = torch.from_numpy(data["body_motion"]).float()
            feature_valid = torch.from_numpy(data["body_feature_valid_mask"]).bool()
            previous_root = (
                torch.from_numpy(data["previous_root_frame"]).float()
                if "previous_root_frame" in data else None
            )
        if root.ndim != 2 or root.shape[-1] != ROOT_DIM:
            raise ValueError(f"root_motion must have shape [F,{ROOT_DIM}] in {path}")
        if body.ndim != 2 or body.shape[-1] != BODY_DIM:
            raise ValueError(f"body_motion must have shape [F,{BODY_DIM}] in {path}")
        if tuple(feature_valid.shape) != tuple(body.shape):
            raise ValueError(f"body_feature_valid_mask must match body_motion in {path}")
        if root.shape[0] != body.shape[0]:
            raise ValueError(f"root_motion and body_motion lengths differ in {path}")
        if root.shape[0] % FRAMES_PER_TOKEN:
            raise ValueError(f"artifact frame length must be divisible by four in {path}")
        if not bool(torch.isfinite(root).all()) or not bool(
            torch.isfinite(body).all()
        ):
            raise ValueError(f"motion artifact contains non-finite values in {path}")
        if previous_root is not None:
            if tuple(previous_root.shape) != (ROOT_DIM,):
                raise ValueError(
                    f"previous_root_frame must have shape [{ROOT_DIM}] in {path}"
                )
            if not bool(torch.isfinite(previous_root).all()):
                raise ValueError(f"previous_root_frame contains non-finite values in {path}")
        available = root.shape[0]
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
        start_token = (
            random.randint(0, max_start_token) if self.split == "train" else 0
        )
        start = start_token * FRAMES_PER_TOKEN
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
                previous_root = rotate_root_yaw(
                    previous_root[None, None], angle
                )[0, 0]
        return {
            "body_motion": body,
            "root_motion": root,
            "body_feature_valid_mask": feature_valid,
            "previous_root_frame": previous_root,
            "name": str(record["name"]),
            "dataset": str(record["dataset"]),
            "text": "",
        }


def collate_humanml3d(batch: list[dict]) -> dict[str, object]:
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


__all__ = [
    "HumanML3DDataset",
    "collate_humanml3d",
    "load_humanml3d_records",
]
