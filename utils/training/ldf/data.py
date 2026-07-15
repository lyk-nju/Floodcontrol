"""Window and context construction for future LDF training.

Source Datasets return complete physical root5/body265 clips.  This module owns
the LDF-specific view of those clips: an active token-aligned window, the fixed
left context required by the body VAE encoder, and captions expressed on the
active token timeline.
"""

from __future__ import annotations

import random

import torch

from utils.motion_process import BODY_DIM, ROOT_DIM, rotate_motion_yaw, rotate_root_yaw
from utils.token_frame import FRAMES_PER_TOKEN


class LDFWindowCollator:
    """Construct padded active windows and fixed body-encoder context."""

    def __init__(
        self,
        *,
        min_frames: int,
        max_frames: int,
        encoder_context_tokens: int,
        training: bool,
        random_yaw: bool = False,
    ):
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.encoder_context_tokens = int(encoder_context_tokens)
        self.context_frames = self.encoder_context_tokens * FRAMES_PER_TOKEN
        self.training = bool(training)
        self.random_yaw = bool(random_yaw and training)
        for name, value in (("min_frames", self.min_frames), ("max_frames", self.max_frames)):
            if value < FRAMES_PER_TOKEN or value % FRAMES_PER_TOKEN:
                raise ValueError(f"{name} must be a positive multiple of four")
        if self.min_frames > self.max_frames:
            raise ValueError("min_frames must not exceed max_frames")
        if self.encoder_context_tokens < 0:
            raise ValueError("encoder_context_tokens must be non-negative")

    def _select_window(self, available: int, identity: str) -> tuple[int, int]:
        if available < self.min_frames:
            raise ValueError(
                f"sample {identity} has {available} frames, fewer than the required "
                f"{self.min_frames}"
            )
        maximum = min(available, self.max_frames) // FRAMES_PER_TOKEN
        if self.training:
            tokens = random.randint(self.min_frames // FRAMES_PER_TOKEN, maximum)
            start_token = random.randint(0, available // FRAMES_PER_TOKEN - tokens)
        else:
            tokens = maximum
            start_token = 0
        return start_token * FRAMES_PER_TOKEN, tokens * FRAMES_PER_TOKEN

    def _active_text(
        self,
        annotations: list[dict[str, object]],
        *,
        start: int,
        frames: int,
    ) -> list[dict[str, object]]:
        grouped: dict[tuple[int, int], list[dict[str, object]]] = {}
        for annotation in annotations:
            interval = (int(annotation["start_frame"]), int(annotation["end_frame"]))
            grouped.setdefault(interval, []).append(annotation)

        active_end = start + frames
        selected: list[dict[str, object]] = []
        for (caption_start, caption_end), alternatives in sorted(grouped.items()):
            overlap_start = max(start, caption_start)
            overlap_end = min(active_end, caption_end)
            if overlap_start >= overlap_end:
                continue
            annotation = random.choice(alternatives) if self.training else alternatives[0]
            relative_start = overlap_start - start
            relative_end = overlap_end - start
            selected.append(
                {
                    "text": str(annotation["text"]),
                    "tokens": list(annotation.get("tokens", [])),
                    "start_token": relative_start // FRAMES_PER_TOKEN,
                    "end_token": (relative_end + FRAMES_PER_TOKEN - 1) // FRAMES_PER_TOKEN,
                }
            )
        return selected

    def _window(self, sample: dict[str, object]) -> dict[str, object]:
        full_root = sample["root_motion"]
        full_body = sample["body_motion"]
        full_feature_valid = sample["body_feature_valid_mask"]
        identity = f"{sample['dataset']}/{sample['name']}"
        start, frames = self._select_window(int(full_root.shape[0]), identity)
        end = start + frames

        context_start = max(0, start - self.context_frames)
        context_body = full_body[context_start:start].clone()
        context_valid = full_feature_valid[context_start:start].clone()
        active_root = full_root[start:end].clone()
        active_body = full_body[start:end].clone()
        active_valid = full_feature_valid[start:end].clone()
        previous = full_root[start - 1].clone() if start > 0 else None

        origin_x, origin_z = active_root[0, 0].clone(), active_root[0, 2].clone()
        active_root[:, 0] -= origin_x
        active_root[:, 2] -= origin_z
        if previous is not None:
            previous[0] -= origin_x
            previous[2] -= origin_z

        if self.random_yaw:
            angle = torch.rand(1, device=active_root.device) * (2.0 * torch.pi)
            joined_body = torch.cat([context_body, active_body], dim=0)
            joined_root = full_root[context_start:end].clone()
            joined_root[:, 0] -= origin_x
            joined_root[:, 2] -= origin_z
            joined_root, joined_body = rotate_motion_yaw(
                joined_root[None], joined_body[None], angle
            )
            context_length = int(context_body.shape[0])
            context_body = joined_body[0, :context_length]
            active_root = joined_root[0, context_length:]
            active_body = joined_body[0, context_length:]
            if previous is not None:
                previous = rotate_root_yaw(previous[None, None], angle)[0, 0]

        missing_context = self.context_frames - int(context_body.shape[0])
        body_with_context = torch.cat(
            [torch.zeros(missing_context, BODY_DIM, dtype=active_body.dtype), context_body, active_body],
            dim=0,
        )
        feature_with_context = torch.cat(
            [
                torch.zeros(missing_context, BODY_DIM, dtype=torch.bool),
                context_valid,
                active_valid,
            ],
            dim=0,
        )
        context_frame_valid = torch.zeros(self.context_frames, dtype=torch.bool)
        if context_body.shape[0]:
            context_frame_valid[missing_context:] = True

        return {
            "dataset": sample["dataset"],
            "name": sample["name"],
            "root_motion": active_root,
            "body_motion": active_body,
            "body_feature_valid_mask": active_valid,
            "body_with_context": body_with_context,
            "body_with_context_feature_valid_mask": feature_with_context,
            "context_frame_valid_mask": context_frame_valid,
            "previous_root_frame": previous,
            "text_data": self._active_text(
                list(sample.get("text_data", [])), start=start, frames=frames
            ),
        }

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        if not samples:
            raise ValueError("LDFWindowCollator requires a non-empty batch")
        windows = [self._window(sample) for sample in samples]
        batch_size = len(windows)
        max_active_frames = max(int(item["body_motion"].shape[0]) for item in windows)
        total_frames = self.context_frames + max_active_frames

        root = torch.zeros(batch_size, max_active_frames, ROOT_DIM)
        body = torch.zeros(batch_size, max_active_frames, BODY_DIM)
        feature_valid = torch.zeros(
            batch_size, max_active_frames, BODY_DIM, dtype=torch.bool
        )
        frame_valid = torch.zeros(batch_size, max_active_frames, dtype=torch.bool)
        body_with_context = torch.zeros(batch_size, total_frames, BODY_DIM)
        context_feature_valid = torch.zeros(
            batch_size, total_frames, BODY_DIM, dtype=torch.bool
        )
        encoder_frame_valid = torch.zeros(batch_size, total_frames, dtype=torch.bool)
        context_frame_valid = torch.zeros(
            batch_size, self.context_frames, dtype=torch.bool
        )
        previous = torch.zeros(batch_size, ROOT_DIM)
        previous_valid = torch.zeros(batch_size, dtype=torch.bool)

        for index, item in enumerate(windows):
            frames = int(item["body_motion"].shape[0])
            root[index, :frames] = item["root_motion"]
            body[index, :frames] = item["body_motion"]
            feature_valid[index, :frames] = item["body_feature_valid_mask"]
            frame_valid[index, :frames] = True
            source = item["body_with_context"]
            body_with_context[index, : self.context_frames + frames] = source
            context_feature_valid[index, : self.context_frames + frames] = item[
                "body_with_context_feature_valid_mask"
            ]
            context_frame_valid[index] = item["context_frame_valid_mask"]
            encoder_frame_valid[index, : self.context_frames] = item[
                "context_frame_valid_mask"
            ]
            encoder_frame_valid[index, self.context_frames : self.context_frames + frames] = True
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
            "context_frame_valid_mask": context_frame_valid,
            "context_token_count": self.encoder_context_tokens,
            "previous_root_frame": previous,
            "previous_root_valid_mask": previous_valid,
            "dataset": [str(item["dataset"]) for item in windows],
            "name": [str(item["name"]) for item in windows],
            "text_data": [item["text_data"] for item in windows],
        }


__all__ = ["LDFWindowCollator"]
