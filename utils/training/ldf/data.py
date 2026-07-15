"""Physical source-span construction for LDF training.

Datasets own complete root5/body265 clips.  This module selects one fixed,
token-aligned physical span for a batch and supplies the real causal body
context needed by the frozen EMA VAE.  H/A/frontier regions, flow noise and
self-forcing state deliberately belong to the later training kernel.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from utils.initialize import instantiate_target
from utils.motion_process import BODY_DIM, ROOT_DIM, rotate_motion_yaw, rotate_root_yaw
from utils.token_frame import FRAMES_PER_TOKEN


class MinimumFrameDataset(Dataset):
    """Task-local view that excludes clips too short for one LDF span."""

    def __init__(self, dataset: Dataset, *, min_frames: int):
        self.min_frames = int(min_frames)
        self.samples: list[tuple[Dataset, int]] = []
        self.rejected_count = 0
        sources = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
        for source in sources:
            records = getattr(source, "dataset", None)
            for index in range(len(source)):
                frame_count = None
                if isinstance(records, list) and index < len(records):
                    record = records[index]
                    motion_path = record.get("motion_path") if isinstance(record, dict) else None
                    if motion_path is not None:
                        with np.load(motion_path, allow_pickle=False) as values:
                            frame_count = int(values["root_motion"].shape[0])
                if frame_count is None:
                    frame_count = int(source[index]["root_motion"].shape[0])
                if frame_count >= self.min_frames:
                    self.samples.append((source, index))
                else:
                    self.rejected_count += 1
        if not self.samples:
            raise RuntimeError(
                f"no dataset samples contain the required {self.min_frames} frames"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        source, source_index = self.samples[index]
        return source[source_index]


class LDFSpanCollator:
    """Crop one batch-shared physical span without constructing diffusion state."""

    def __init__(
        self,
        *,
        min_frames: int,
        max_frames: int,
        encoder_context_tokens: int,
        training: bool,
        random_yaw: bool = False,
        cold_start_probability: float = 0.1,
        cold_start: bool | None = None,
    ):
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.encoder_context_tokens = int(encoder_context_tokens)
        self.context_frames = self.encoder_context_tokens * FRAMES_PER_TOKEN
        self.training = bool(training)
        self.random_yaw = bool(random_yaw and training)
        self.cold_start_probability = float(cold_start_probability)
        self.cold_start = cold_start

        for name, value in (
            ("min_frames", self.min_frames),
            ("max_frames", self.max_frames),
        ):
            if value < FRAMES_PER_TOKEN or value % FRAMES_PER_TOKEN:
                raise ValueError(f"{name} must be a positive multiple of four")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must not exceed max_frames")
        if self.encoder_context_tokens < 0:
            raise ValueError("encoder_context_tokens must be non-negative")
        if not 0.0 <= self.cold_start_probability <= 1.0:
            raise ValueError("cold_start_probability must lie in [0,1]")

    def _select_cold_start(self) -> bool:
        if self.cold_start is not None:
            return bool(self.cold_start)
        if not self.training:
            return True
        return random.random() < self.cold_start_probability

    def _select_span_tokens(self, samples: list[dict[str, object]]) -> int:
        available = min(
            int(sample["root_motion"].shape[0]) // FRAMES_PER_TOKEN
            for sample in samples
        )
        minimum = self.min_frames // FRAMES_PER_TOKEN
        maximum = min(available, self.max_frames // FRAMES_PER_TOKEN)
        if maximum < minimum:
            identities = ", ".join(
                f"{sample['dataset']}/{sample['name']}" for sample in samples
            )
            raise ValueError(
                f"LDF span requires at least {self.min_frames} frames; "
                f"short batch contains {identities}"
            )
        return random.randint(minimum, maximum) if self.training else maximum

    def _select_start_token(
        self,
        *,
        available_tokens: int,
        span_tokens: int,
        cold_start: bool,
    ) -> int:
        if cold_start:
            return 0
        maximum = available_tokens - span_tokens
        if self.training:
            return random.randint(0, maximum)
        return maximum // 2

    def _select_caption(
        self,
        alternatives: list[dict[str, object]],
    ) -> dict[str, object]:
        return random.choice(alternatives) if self.training else alternatives[0]

    def _prompt_timeline(
        self,
        dataset: str,
        annotations: list[dict[str, object]],
        *,
        start_frame: int,
        span_frames: int,
    ) -> list[str]:
        """Compile source captions into one prompt for every motion token.

        HumanML3D describes one action clip, so one relevant caption is chosen
        and repeated across the complete sampled span.  BABEL supplies a
        piecewise action timeline; each four-frame token receives only the
        caption whose interval owns that complete token.
        """

        span_tokens = span_frames // FRAMES_PER_TOKEN
        if not annotations:
            return [""] * span_tokens

        grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
        for annotation in annotations:
            interval = (
                int(annotation["start_frame"]),
                int(annotation["end_frame"]),
            )
            if interval[0] < 0 or interval[1] <= interval[0]:
                raise ValueError("text intervals must be positive half-open ranges")
            grouped.setdefault(interval, []).append(annotation)

        span_end = start_frame + span_frames
        if dataset == "HumanML3D":
            candidates = []
            for (caption_start, caption_end), alternatives in grouped.items():
                overlap = max(
                    0,
                    min(span_end, caption_end) - max(start_frame, caption_start),
                )
                if overlap:
                    covers_span = caption_start <= start_frame and caption_end >= span_end
                    candidates.append(
                        (
                            int(covers_span),
                            overlap,
                            caption_end - caption_start,
                            -caption_start,
                            alternatives,
                        )
                    )
            if not candidates:
                return [""] * span_tokens
            alternatives = max(candidates, key=lambda item: item[:-1])[-1]
            text = str(self._select_caption(alternatives)["text"])
            return [text] * span_tokens

        timeline = [""] * span_tokens
        for (caption_start, caption_end), alternatives in sorted(grouped.items()):
            overlap_start = max(start_frame, caption_start)
            overlap_end = min(span_end, caption_end)
            if overlap_start >= overlap_end:
                continue
            if caption_start % FRAMES_PER_TOKEN or caption_end % FRAMES_PER_TOKEN:
                raise ValueError(
                    f"{dataset} text intervals must align to four-frame tokens"
                )
            annotation = self._select_caption(alternatives)
            first = max(0, (caption_start - start_frame) // FRAMES_PER_TOKEN)
            last = min(span_tokens, (caption_end - start_frame) // FRAMES_PER_TOKEN)
            text = str(annotation["text"])
            for token_index in range(first, last):
                if timeline[token_index] and timeline[token_index] != text:
                    raise ValueError(
                        f"{dataset} text intervals overlap at token {token_index}"
                    )
                timeline[token_index] = text
        return timeline

    def _crop_sample(
        self,
        sample: dict[str, object],
        *,
        span_tokens: int,
        cold_start: bool,
    ) -> dict[str, object]:
        full_root = sample["root_motion"]
        full_body = sample["body_motion"]
        full_feature_valid = sample["body_feature_valid_mask"]
        available_tokens = int(full_root.shape[0]) // FRAMES_PER_TOKEN
        start_token = self._select_start_token(
            available_tokens=available_tokens,
            span_tokens=span_tokens,
            cold_start=cold_start,
        )
        start = start_token * FRAMES_PER_TOKEN
        frames = span_tokens * FRAMES_PER_TOKEN
        end = start + frames

        context_tokens = 0 if cold_start else min(
            start_token, self.encoder_context_tokens
        )
        context_start = start - context_tokens * FRAMES_PER_TOKEN
        joined_root = full_root[context_start:end].clone()
        joined_body = full_body[context_start:end].clone()
        joined_valid = full_feature_valid[context_start:end].clone()
        previous = full_root[start - 1].clone() if start > 0 and not cold_start else None

        if self.random_yaw:
            angle = torch.rand(1, device=joined_root.device) * (2.0 * torch.pi)
            joined_root, joined_body = rotate_motion_yaw(
                joined_root[None], joined_body[None], angle
            )
            joined_root = joined_root[0]
            joined_body = joined_body[0]
            if previous is not None:
                previous = rotate_root_yaw(previous[None, None], angle)[0, 0]

        context_frames = context_tokens * FRAMES_PER_TOKEN
        return {
            "dataset": sample["dataset"],
            "name": sample["name"],
            "root_motion": joined_root[context_frames:],
            "body_motion": joined_body[context_frames:],
            "body_feature_valid_mask": joined_valid[context_frames:],
            "body_with_context": joined_body,
            "body_with_context_feature_valid_mask": joined_valid,
            "context_token_count": context_tokens,
            "previous_root_frame": previous,
            "source_start_token": start_token,
            "prompt_timeline": self._prompt_timeline(
                str(sample["dataset"]),
                list(sample.get("text_data", [])),
                start_frame=start,
                span_frames=frames,
            ),
        }

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        if not samples:
            raise ValueError("LDFSpanCollator requires a non-empty batch")
        cold_start = self._select_cold_start()
        span_tokens = self._select_span_tokens(samples)
        spans = [
            self._crop_sample(
                sample,
                span_tokens=span_tokens,
                cold_start=cold_start,
            )
            for sample in samples
        ]
        batch_size = len(spans)
        span_frames = span_tokens * FRAMES_PER_TOKEN
        total_frames = max(int(item["body_with_context"].shape[0]) for item in spans)

        root = torch.zeros(batch_size, span_frames, ROOT_DIM)
        body = torch.zeros(batch_size, span_frames, BODY_DIM)
        feature_valid = torch.zeros(
            batch_size, span_frames, BODY_DIM, dtype=torch.bool
        )
        frame_valid = torch.ones(batch_size, span_frames, dtype=torch.bool)
        body_with_context = torch.zeros(batch_size, total_frames, BODY_DIM)
        context_feature_valid = torch.zeros(
            batch_size, total_frames, BODY_DIM, dtype=torch.bool
        )
        encoder_frame_valid = torch.zeros(batch_size, total_frames, dtype=torch.bool)
        context_token_count = torch.zeros(batch_size, dtype=torch.long)
        previous = torch.zeros(batch_size, ROOT_DIM)
        previous_valid = torch.zeros(batch_size, dtype=torch.bool)
        source_start = torch.zeros(batch_size, dtype=torch.long)

        for index, item in enumerate(spans):
            root[index] = item["root_motion"]
            body[index] = item["body_motion"]
            feature_valid[index] = item["body_feature_valid_mask"]
            source = item["body_with_context"]
            encoder_frames = int(source.shape[0])
            body_with_context[index, :encoder_frames] = source
            context_feature_valid[index, :encoder_frames] = item[
                "body_with_context_feature_valid_mask"
            ]
            encoder_frame_valid[index, :encoder_frames] = True
            context_token_count[index] = int(item["context_token_count"])
            source_start[index] = int(item["source_start_token"])
            if item["previous_root_frame"] is not None:
                previous[index] = item["previous_root_frame"]
                previous_valid[index] = True

        return {
            "root_motion": root,
            "body_motion": body,
            "body_feature_valid_mask": feature_valid,
            "frame_valid_mask": frame_valid,
            "body_with_context": body_with_context,
            "body_with_context_feature_valid_mask": context_feature_valid,
            "body_with_context_frame_valid_mask": encoder_frame_valid,
            "context_token_count": context_token_count,
            "previous_root_frame": previous,
            "previous_root_valid_mask": previous_valid,
            "source_start_token": source_start,
            "cold_start_mask": torch.full(
                (batch_size,), cold_start, dtype=torch.bool
            ),
            "span_token_count": span_tokens,
            "dataset": [str(item["dataset"]) for item in spans],
            "name": [str(item["name"]) for item in spans],
            "prompt_timeline": [item["prompt_timeline"] for item in spans],
        }


def create_dataset(cfg, split: str):
    common_args = {"split": split, "fps": float(cfg.model.params.fps)}
    dataset_configs = cfg.data.get("datasets", None)
    if dataset_configs:
        dataset = instantiate_target(
            cfg.data.target,
            cfg=None,
            dataset_configs=dataset_configs,
            **common_args,
        )
        return MinimumFrameDataset(dataset, min_frames=int(cfg.data.min_frames))
    meta_paths = cfg.data.get(f"{split}_meta_paths", None)
    if not meta_paths:
        raise RuntimeError(f"set data.{split}_meta_paths to processed motion splits")
    dataset = instantiate_target(
        cfg.data.target,
        cfg=None,
        meta_paths=meta_paths,
        artifact_path=cfg.data.get("artifact_path", "artifacts"),
        text_path=cfg.data.get("text_path"),
        **common_args,
    )
    return MinimumFrameDataset(dataset, min_frames=int(cfg.data.min_frames))


def create_dataloaders(
    cfg,
    *,
    encoder_context_tokens: int,
) -> tuple[DataLoader | None, DataLoader]:
    train_dataset = create_dataset(cfg, "train") if cfg.train else None
    val_dataset = create_dataset(cfg, "val")
    common = {
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
    }
    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=int(cfg.data.train_batch_size),
            shuffle=True,
            collate_fn=LDFSpanCollator(
                min_frames=cfg.data.min_frames,
                max_frames=cfg.data.max_frames,
                encoder_context_tokens=encoder_context_tokens,
                training=True,
                random_yaw=cfg.data.random_yaw,
                cold_start_probability=cfg.data.cold_start_probability,
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
        collate_fn=LDFSpanCollator(
            min_frames=cfg.data.min_frames,
            max_frames=cfg.data.max_frames,
            encoder_context_tokens=encoder_context_tokens,
            training=False,
            random_yaw=False,
            cold_start=True,
        ),
        **common,
    )
    return train_loader, val_loader


__all__ = [
    "LDFSpanCollator",
    "MinimumFrameDataset",
    "create_dataloaders",
    "create_dataset",
]
