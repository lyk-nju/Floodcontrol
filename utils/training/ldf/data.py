"""Physical source-span construction for LDF training.

Datasets own complete root5/body265 clips.  This module selects one fixed,
token-aligned physical span for a batch and supplies the real causal body
context needed by the frozen EMA VAE.  H/A/frontier regions, flow noise and
self-forcing state deliberately belong to the later training kernel.
"""

from __future__ import annotations

import random
from collections import defaultdict
from math import ceil

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from utils.initialize import instantiate_target
from utils.motion_process import BODY_DIM, ROOT_DIM, rotate_motion_yaw, rotate_root_yaw
from utils.token_frame import FRAMES_PER_TOKEN


_SEED_MASK = (1 << 64) - 1


def _mix_seed(values: list[int]) -> int:
    """Combine ordered integer seeds without the collisions of a weighted sum."""

    state = 0x6A09E667F3BCC909
    for value in values:
        mixed = (int(value) & _SEED_MASK) + 0x9E3779B97F4A7C15
        mixed &= _SEED_MASK
        mixed ^= mixed >> 30
        mixed = (mixed * 0xBF58476D1CE4E5B9) & _SEED_MASK
        mixed ^= mixed >> 27
        mixed = (mixed * 0x94D049BB133111EB) & _SEED_MASK
        mixed ^= mixed >> 31
        state ^= (
            mixed
            + 0x9E3779B97F4A7C15
            + ((state << 6) & _SEED_MASK)
            + (state >> 2)
        ) & _SEED_MASK
        state &= _SEED_MASK
    return state % (2**63 - 1)


class MinimumFrameDataset(Dataset):
    """Task-local view that excludes clips too short for one LDF span."""

    def __init__(self, dataset: Dataset, *, min_frames: int):
        self.min_frames = int(min_frames)
        self.samples: list[tuple[Dataset, int]] = []
        self.frame_counts: list[int] = []
        self.rejected_count = 0
        sources = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
        for source in sources:
            records = getattr(source, "dataset", None)
            known_frame_counts = getattr(source, "frame_counts", None)
            for index in range(len(source)):
                frame_count = None
                if (
                    isinstance(known_frame_counts, list)
                    and index < len(known_frame_counts)
                ):
                    frame_count = int(known_frame_counts[index])
                if (
                    frame_count is None
                    and isinstance(records, list)
                    and index < len(records)
                ):
                    record = records[index]
                    motion_path = record.get("motion_path") if isinstance(record, dict) else None
                    if motion_path is not None:
                        with np.load(motion_path, allow_pickle=False) as values:
                            frame_count = int(values["root_motion"].shape[0])
                if frame_count is None:
                    frame_count = int(source[index]["root_motion"].shape[0])
                if frame_count >= self.min_frames:
                    self.samples.append((source, index))
                    self.frame_counts.append(frame_count)
                else:
                    self.rejected_count += 1
        if not self.samples:
            raise RuntimeError(
                f"no dataset samples contain the required {self.min_frames} frames"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, int]):
        augmentation_seed = None
        if isinstance(index, tuple):
            index, augmentation_seed = index
        source, source_index = self.samples[index]
        sample = dict(source[source_index])
        # This task-local index is stable across workers and lets validation
        # distribute deterministic probes without changing Dataset contracts.
        sample["_ldf_sample_index"] = int(index)
        if augmentation_seed is not None:
            sample["_augmentation_seed"] = int(augmentation_seed)
        return sample


class LengthBucketBatchSampler(Sampler[list[tuple[int, int]]]):
    """Shuffle clips in length buckets before forming training batches."""

    def __init__(
        self,
        dataset: MinimumFrameDataset,
        *,
        batch_size: int,
        bucket_width_frames: int,
        max_frames: int,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.bucket_width_frames = int(bucket_width_frames)
        self.max_frames = int(max_frames)
        self.seed = int(seed)
        self.epoch = 0
        # Lightning discovers set_epoch() through batch_sampler.sampler.
        self.sampler = self
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if (
            self.bucket_width_frames < FRAMES_PER_TOKEN
            or self.bucket_width_frames % FRAMES_PER_TOKEN
        ):
            raise ValueError("bucket_width_frames must be a positive multiple of four")

        self.buckets: dict[int, list[int]] = defaultdict(list)
        for index, frame_count in enumerate(dataset.frame_counts):
            capped = min(int(frame_count), self.max_frames)
            self.buckets[capped // self.bucket_width_frames].append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches: list[list[int]] = []
        for indices in self.buckets.values():
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[position] for position in order]
            batches.extend(
                shuffled[start : start + self.batch_size]
                for start in range(0, len(shuffled), self.batch_size)
            )
        order = torch.randperm(len(batches), generator=generator).tolist()
        epoch = self.epoch
        for position in order:
            yield [
                (index, self.seed + epoch * 1_000_003 + index)
                for index in batches[position]
            ]

    def __len__(self) -> int:
        return sum(
            ceil(len(indices) / self.batch_size) for indices in self.buckets.values()
        )


class ResumableDataLoader(DataLoader):
    """Expose consumed-batch state to Lightning's CombinedLoader checkpoint."""

    _STATE_VERSION = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._active_epoch: int | None = None
        self._yielded_batches = 0

    def __iter__(self):
        sampler = self.batch_sampler
        epoch = int(getattr(sampler, "epoch", 0))
        if self._active_epoch != epoch:
            self._active_epoch = epoch
            self._yielded_batches = 0

        iterator = super().__iter__()
        skipped = 0
        while skipped < self._yielded_batches:
            try:
                next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "saved LDF dataloader cursor exceeds the reconstructed epoch"
                ) from error
            skipped += 1

        for batch in iterator:
            self._yielded_batches += 1
            yield batch

    def state_dict(self) -> dict[str, int]:
        epoch = int(getattr(self.batch_sampler, "epoch", 0))
        return {
            "version": self._STATE_VERSION,
            "epoch": epoch,
            "yielded_batches": int(self._yielded_batches),
        }

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        if int(state_dict.get("version", -1)) != self._STATE_VERSION:
            raise RuntimeError("unsupported LDF dataloader checkpoint state")
        epoch = int(state_dict["epoch"])
        yielded_batches = int(state_dict["yielded_batches"])
        if epoch < 0 or not 0 <= yielded_batches <= len(self):
            raise RuntimeError("invalid LDF dataloader checkpoint cursor")
        set_epoch = getattr(self.batch_sampler, "set_epoch", None)
        if not callable(set_epoch):
            raise RuntimeError("resumable LDF dataloader requires an epoch-aware sampler")
        set_epoch(epoch)
        self._active_epoch = epoch
        self._yielded_batches = yielded_batches


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
        validation_probe: str | None = None,
        validation_span_frames: int | None = None,
        validation_positions: tuple[str, ...] | None = None,
    ):
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)
        self.encoder_context_tokens = int(encoder_context_tokens)
        self.context_frames = self.encoder_context_tokens * FRAMES_PER_TOKEN
        self.training = bool(training)
        self.random_yaw = bool(random_yaw and training)
        self.cold_start_probability = float(cold_start_probability)
        self.cold_start = cold_start
        self.validation_probe = validation_probe
        self.validation_span_frames = (
            None if validation_span_frames is None else int(validation_span_frames)
        )
        self.validation_positions = validation_positions

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
        if self.validation_span_frames is not None:
            if self.training:
                raise ValueError("validation_span_frames is only valid for validation")
            if (
                self.validation_span_frames < self.min_frames
                or self.validation_span_frames > self.max_frames
                or self.validation_span_frames % FRAMES_PER_TOKEN
            ):
                raise ValueError(
                    "validation_span_frames must be a four-frame multiple within "
                    "[min_frames,max_frames]"
                )
        if self.validation_positions is not None:
            if self.training or self.cold_start is not False:
                raise ValueError(
                    "validation_positions requires a continuation validation collator"
                )
            allowed = {"early", "middle", "late"}
            if not self.validation_positions or not set(
                self.validation_positions
            ).issubset(allowed):
                raise ValueError(
                    "validation_positions may only contain early/middle/late"
                )

    def _select_cold_start(self, rng=random) -> bool:
        if self.cold_start is not None:
            return bool(self.cold_start)
        if not self.training:
            return True
        return rng.random() < self.cold_start_probability

    def _select_span_tokens(
        self, samples: list[dict[str, object]], rng=random
    ) -> int:
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
        if self.training:
            return rng.randint(minimum, maximum)
        if self.validation_span_frames is not None:
            requested = self.validation_span_frames // FRAMES_PER_TOKEN
            if requested > maximum:
                raise ValueError(
                    "validation sample is shorter than validation_span_frames"
                )
            return requested
        return maximum

    def _select_start_token(
        self,
        *,
        available_tokens: int,
        span_tokens: int,
        cold_start: bool,
        validation_position: str | None = None,
        rng=random,
    ) -> int:
        if cold_start:
            return 0
        maximum = available_tokens - span_tokens
        if self.training:
            return rng.randint(0, maximum)
        if maximum <= 0:
            raise ValueError(
                "continuation validation requires at least one source token before "
                "the active span"
            )
        if validation_position == "early":
            return 1
        if validation_position == "late":
            return maximum
        return max(1, maximum // 2)

    def _select_caption(
        self,
        alternatives: list[dict[str, object]],
        rng=random,
    ) -> dict[str, object]:
        return rng.choice(alternatives) if self.training else alternatives[0]

    def _prompt_timeline(
        self,
        dataset: str,
        annotations: list[dict[str, object]],
        *,
        start_frame: int,
        span_frames: int,
        rng=random,
    ) -> list[str]:
        """Compile source captions into one prompt for every motion token.

        HumanML3D describes one action clip, so one relevant caption is chosen
        and repeated across the complete sampled span. BABEL annotations may
        end at arbitrary frames, so each four-frame token chooses the caption
        with the largest frame overlap.
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
            text = str(self._select_caption(alternatives, rng)["text"])
            return [text] * span_tokens

        intervals = [
            (caption_start, caption_end, order, alternatives)
            for order, ((caption_start, caption_end), alternatives) in enumerate(
                grouped.items()
            )
        ]
        timeline = []
        for token_index in range(span_tokens):
            token_start = start_frame + token_index * FRAMES_PER_TOKEN
            token_end = token_start + FRAMES_PER_TOKEN
            candidates = []
            for caption_start, caption_end, order, alternatives in intervals:
                overlap = max(
                    0,
                    min(token_end, caption_end) - max(token_start, caption_start),
                )
                if overlap:
                    candidates.append(
                        (
                            overlap,
                            -(caption_end - caption_start),
                            -order,
                            alternatives,
                        )
                    )
            if not candidates:
                timeline.append("")
                continue
            alternatives = max(candidates, key=lambda item: item[:-1])[-1]
            timeline.append(str(self._select_caption(alternatives, rng)["text"]))
        return timeline

    def _crop_sample(
        self,
        sample: dict[str, object],
        *,
        span_tokens: int,
        cold_start: bool,
        validation_position: str | None = None,
        rng=random,
        torch_generator: torch.Generator | None = None,
    ) -> dict[str, object]:
        full_root = sample["root_motion"]
        full_body = sample["body_motion"]
        full_feature_valid = sample["body_feature_valid_mask"]
        available_tokens = int(full_root.shape[0]) // FRAMES_PER_TOKEN
        start_token = self._select_start_token(
            available_tokens=available_tokens,
            span_tokens=span_tokens,
            cold_start=cold_start,
            validation_position=validation_position,
            rng=rng,
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
            angle = torch.rand(
                1,
                device=joined_root.device,
                generator=torch_generator,
            ) * (2.0 * torch.pi)
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
                rng=rng,
            ),
        }

    def __call__(self, samples: list[dict[str, object]]) -> dict[str, object]:
        if not samples:
            raise ValueError("LDFSpanCollator requires a non-empty batch")
        seeds = [sample.get("_augmentation_seed") for sample in samples]
        if any(seed is not None for seed in seeds) and not all(
            seed is not None for seed in seeds
        ):
            raise ValueError("training samples must either all carry seeds or all omit them")
        if all(seed is not None for seed in seeds):
            batch_seed = _mix_seed([int(seed) for seed in seeds])
            rng = random.Random(batch_seed)
            torch_generator = torch.Generator().manual_seed(batch_seed)
        else:
            rng = random
            torch_generator = None
        cold_start = self._select_cold_start(rng)
        span_tokens = self._select_span_tokens(samples, rng)
        spans = []
        for sample in samples:
            validation_position = None
            if self.validation_positions is not None:
                sample_index = int(sample.get("_ldf_sample_index", 0))
                validation_position = self.validation_positions[
                    sample_index % len(self.validation_positions)
                ]
            span = self._crop_sample(
                sample,
                span_tokens=span_tokens,
                cold_start=cold_start,
                validation_position=validation_position,
                rng=rng,
                torch_generator=torch_generator,
            )
            span["validation_position"] = validation_position
            spans.append(span)
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

        output = {
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
        if self.validation_probe is not None:
            output["validation_probe"] = self.validation_probe
        if self.validation_positions is not None:
            output["validation_position"] = [
                item["validation_position"] for item in spans
            ]
        return output


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
) -> tuple[DataLoader | None, list[DataLoader]]:
    train_dataset = create_dataset(cfg, "train") if cfg.train else None
    val_dataset = create_dataset(cfg, "val")
    common = {
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": bool(cfg.data.get("pin_memory", True)),
    }
    train_loader = None
    if train_dataset is not None:
        batch_sampler = LengthBucketBatchSampler(
            train_dataset,
            batch_size=int(cfg.data.train_batch_size),
            bucket_width_frames=int(cfg.data.get("length_bucket_frames", 20)),
            max_frames=int(cfg.data.max_frames),
            seed=int(cfg.get("seed", 0)),
        )
        train_loader = ResumableDataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
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

    validation = cfg.get("validation") or {}
    continuation_span_frames = int(
        validation.get("continuation_span_frames", cfg.data.min_frames)
    )
    if continuation_span_frames % FRAMES_PER_TOKEN:
        raise ValueError("validation.continuation_span_frames must align to four frames")
    continuation_dataset = MinimumFrameDataset(
        val_dataset,
        min_frames=continuation_span_frames + FRAMES_PER_TOKEN,
    )

    def validation_loader(
        name: str,
        *,
        cold_start: bool,
        dataset: Dataset,
        span_frames: int | None = None,
        positions: tuple[str, ...] | None = None,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=int(cfg.data.val_batch_size),
            shuffle=False,
            collate_fn=LDFSpanCollator(
                min_frames=cfg.data.min_frames,
                max_frames=cfg.data.max_frames,
                encoder_context_tokens=encoder_context_tokens,
                training=False,
                random_yaw=False,
                cold_start=cold_start,
                validation_probe=name,
                validation_span_frames=span_frames,
                validation_positions=positions,
            ),
            **common,
        )

    val_loaders = [
        validation_loader("teacher_cold", cold_start=True, dataset=val_dataset),
        validation_loader(
            "teacher_continuation",
            cold_start=False,
            dataset=continuation_dataset,
            span_frames=continuation_span_frames,
            positions=("early", "middle", "late"),
        ),
    ]
    self_forcing = cfg.get("self_forcing")
    if self_forcing is not None and bool(self_forcing.get("enabled", False)):
        val_loaders.append(
            validation_loader(
                "self_forcing",
                cold_start=False,
                dataset=continuation_dataset,
                span_frames=continuation_span_frames,
                positions=("early", "middle", "late"),
            )
        )
    return train_loader, val_loaders


__all__ = [
    "LDFSpanCollator",
    "LengthBucketBatchSampler",
    "MinimumFrameDataset",
    "ResumableDataLoader",
    "create_dataloaders",
    "create_dataset",
]
