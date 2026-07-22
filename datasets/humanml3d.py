"""HumanML3D Dataset for complete root5/body259 motion sequences."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from torch.utils.data import Dataset
from utils.motion_process import BODY_DIM, ROOT_DIM
from utils.token_frame import FRAMES_PER_TOKEN, MOTION_FPS


SUPPORTED_SPLITS = frozenset({"train", "val", "test"})
DATASET_NAME = "HumanML3D"


class HumanML3DDataset(Dataset):
    """Load complete HumanML3D samples without training-time transformations."""

    def __init__(
        self,
        *,
        meta_paths: Iterable[str | Path],
        split: str,
        artifact_path: str = "artifacts",
        text_path: str | Path | None = None,
        fps: float = MOTION_FPS,
    ):
        self.split = str(split)
        if self.split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split {self.split!r}")

        self.file_list = [Path(path) for path in meta_paths]
        self.motion_path = Path(artifact_path)
        if self.motion_path.is_absolute() or ".." in self.motion_path.parts:
            raise ValueError("artifact_path must be a relative directory")
        self.text_path = None if text_path is None else Path(text_path)
        self.fps = float(fps)
        self.dataset = self._load_file_list()

    def _load_file_list(self) -> list[dict[str, object]]:
        """Build a lightweight index from HumanML3D split TXT files."""

        dataset: list[dict[str, object]] = []
        sample_identities: set[tuple[str, str]] = set()

        for meta_path in self.file_list:
            if not meta_path.is_file():
                raise RuntimeError(f"HumanML3D split file not found at {meta_path}")

            data_path = meta_path.parent
            text_root = self.text_path
            if text_root is not None and not text_root.is_absolute():
                text_root = data_path / text_root

            for line_number, line in enumerate(
                meta_path.read_text().splitlines(), start=1
            ):
                name = line.strip()
                if not name:
                    continue
                if Path(name).name != name:
                    raise ValueError(
                        f"sample id must be a filename stem: "
                        f"{meta_path}:{line_number}"
                    )

                identity = (DATASET_NAME, name)
                if identity in sample_identities:
                    raise ValueError(f"duplicate sample id: {DATASET_NAME}/{name}")
                sample_identities.add(identity)

                data: dict[str, object] = {
                    "dataset": DATASET_NAME,
                    "name": name,
                }

                ##############################
                # motion
                ##############################
                motion_file = data_path / self.motion_path / f"{name}.npz"
                if not motion_file.is_file():
                    raise RuntimeError(
                        f"HumanML3D motion file not found at {motion_file}"
                    )
                data["motion_path"] = motion_file

                ##############################
                # text
                ##############################
                if text_root is not None:
                    text_file = text_root / f"{name}.txt"
                    if not text_file.is_file():
                        raise RuntimeError(
                            f"HumanML3D text file not found at {text_file}"
                        )
                    data["text_path"] = text_file

                dataset.append(data)

        if len(dataset) == 0:
            raise RuntimeError("HumanML3D split contains no samples")
        return dataset

    def load_motion(
        self, motion_path: str | Path
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load one complete physical motion sequence."""

        path = Path(motion_path)
        with np.load(path, allow_pickle=False) as data:
            required_fields = (
                "root_motion",
                "body_motion",
                "body_feature_valid_mask",
            )
            missing_fields = [name for name in required_fields if name not in data]
            if missing_fields:
                raise ValueError(
                    f"motion file {path} is missing {', '.join(missing_fields)}"
                )

            # Ignore optional metadata stored by older preprocessing versions.
            root_array = np.asarray(data["root_motion"])
            body_array = np.asarray(data["body_motion"])
            mask_array = np.asarray(data["body_feature_valid_mask"])
            if root_array.dtype != np.float32 or body_array.dtype != np.float32:
                raise ValueError(f"motion tensors must be float32 in {path}")
            if mask_array.dtype != np.bool_:
                raise ValueError(f"body_feature_valid_mask must be bool in {path}")
            root_motion = torch.from_numpy(root_array)
            body_motion = torch.from_numpy(body_array)
            body_feature_valid_mask = torch.from_numpy(mask_array)

        if root_motion.ndim != 2 or root_motion.shape[-1] != ROOT_DIM:
            raise ValueError(f"root_motion must be [F,{ROOT_DIM}] in {path}")
        if body_motion.ndim != 2 or body_motion.shape[-1] != BODY_DIM:
            raise ValueError(f"body_motion must be [F,{BODY_DIM}] in {path}")
        if (
            root_motion.shape[0] != body_motion.shape[0]
            or body_feature_valid_mask.shape != body_motion.shape
        ):
            raise ValueError(f"motion fields have incompatible shapes in {path}")
        if (
            root_motion.shape[0] < FRAMES_PER_TOKEN
            or root_motion.shape[0] % FRAMES_PER_TOKEN
        ):
            raise ValueError(
                f"motion length must be a positive multiple of four in {path}"
            )
        if not bool(torch.isfinite(root_motion).all()) or not bool(
            torch.isfinite(body_motion).all()
        ):
            raise ValueError(f"motion contains non-finite values in {path}")
        heading_norm = root_motion[:, 3:5].norm(dim=-1)
        if not bool(torch.allclose(heading_norm, torch.ones_like(heading_norm), atol=1e-5)):
            raise ValueError(f"root heading must be normalized in {path}")

        return root_motion, body_motion, body_feature_valid_mask

    def load_text(
        self,
        text_path: str | Path,
        motion_length: int,
    ) -> list[dict[str, object]]:
        """Load HumanML3D captions and convert time tags to frame intervals."""

        text_data: list[dict[str, object]] = []
        path = Path(text_path)

        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue

            line_split = line.split("#")
            if len(line_split) != 4:
                raise ValueError(
                    f"invalid HumanML3D text row at {path}:{line_number}"
                )

            caption = line_split[0]
            text_tokens = line_split[1].split()
            from_tag = float(line_split[2])
            to_tag = float(line_split[3])
            if not math.isfinite(from_tag) or not math.isfinite(to_tag):
                from_tag = 0.0
                to_tag = 0.0

            # In HumanML3D, 0#0 marks a caption for the complete motion.
            if from_tag == 0.0 and to_tag == 0.0:
                start_frame = 0
                end_frame = motion_length
            else:
                # A few source rows have zero-duration or reversed tags. They
                # describe no usable part of the motion and must not be turned
                # into a synthetic interval by clamping or swapping endpoints.
                if to_tag <= from_tag:
                    continue
                start_frame = int(from_tag * self.fps + 0.5)
                end_frame = int(to_tag * self.fps + 0.5)
                start_frame = max(0, min(motion_length, start_frame))
                end_frame = max(0, min(motion_length, end_frame))
                if end_frame <= start_frame:
                    continue

            text_data.append(
                {
                    "text": caption,
                    "tokens": text_tokens,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )

        return text_data

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self._process(self.dataset[index])

    def _process(self, data: dict[str, object]) -> dict[str, object]:
        """Load and assemble one complete sample for task-specific collators."""

        output: dict[str, object] = {
            "dataset": data["dataset"],
            "name": data["name"],
        }

        ##############################
        # motion
        ##############################
        root_motion, body_motion, body_feature_valid_mask = self.load_motion(
            data["motion_path"]
        )
        output["root_motion"] = root_motion
        output["body_motion"] = body_motion
        output["body_feature_valid_mask"] = body_feature_valid_mask

        ##############################
        # text
        ##############################
        if "text_path" in data:
            output["text_data"] = self.load_text(
                data["text_path"], motion_length=int(root_motion.shape[0])
            )
        else:
            output["text_data"] = []

        return output


__all__ = ["HumanML3DDataset"]
