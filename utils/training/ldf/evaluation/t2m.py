"""Canonical batched T2M generation and metric evaluation."""

from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch

from metrics.humanml import convert_root5_body259_to_humanml263
from metrics.t2m import T2MMetrics

from .generation import generate_t2m_evaluation_batch


@dataclass(frozen=True)
class T2MEvaluationBatch:
    """One fixed-length batch with stable global sample indices."""

    frame_count: int
    samples: tuple[tuple[int, dict[str, object]], ...]


def stable_t2m_seed(base_seed: int, *parts: object) -> int:
    """Derive one process-independent seed from the frozen T2M identity."""

    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(base_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little") % (2**63 - 1)


def t2m_frame_count(sample: dict[str, object], maximum: int) -> int:
    """Return the evaluator-visible frame count on the four-frame token grid."""

    frames = min(int(sample["root_motion"].shape[0]), int(maximum))
    frames -= frames % 4
    if frames <= 0:
        raise ValueError("T2M sample has no complete four-frame token")
    return frames


def build_t2m_evaluation_batches(
    samples: Iterable[tuple[int, dict[str, object]]],
    *,
    maximum_frames: int,
    batch_size: int,
) -> tuple[T2MEvaluationBatch, ...]:
    """Build deterministic exact-length batches before distributed sharding."""

    if int(batch_size) <= 0:
        raise ValueError("T2M batch_size must be positive")
    buckets: dict[int, list[tuple[int, dict[str, object]]]] = {}
    for sample_index, sample in samples:
        frames = t2m_frame_count(sample, maximum_frames)
        buckets.setdefault(frames, []).append((int(sample_index), sample))
    batches: list[T2MEvaluationBatch] = []
    for frames in sorted(buckets, reverse=True):
        values = buckets[frames]
        for start in range(0, len(values), int(batch_size)):
            batches.append(
                T2MEvaluationBatch(
                    frame_count=int(frames),
                    samples=tuple(values[start : start + int(batch_size)]),
                )
            )
    return tuple(batches)


def shard_t2m_evaluation_batches(
    batches: Iterable[T2MEvaluationBatch],
    *,
    rank: int,
    world_size: int,
) -> tuple[T2MEvaluationBatch, ...]:
    """Assign complete global batches so world size does not change batch mates."""

    if int(world_size) <= 0:
        raise ValueError("T2M world_size must be positive")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError("T2M rank must be in [0, world_size)")
    return tuple(
        batch
        for batch_index, batch in enumerate(batches)
        if batch_index % int(world_size) == int(rank)
    )


def evaluate_t2m_batches(
    module,
    *,
    metric_config,
    batches: Iterable[T2MEvaluationBatch],
    guidance_mode: str,
    cfg_scale_joint: float,
    base_seed: int,
    generation_mode: str,
    num_denoise_steps: int,
    sanity_flag: bool = False,
    progress_callback: Callable[[int, int, int, float], None] | None = None,
) -> tuple[dict[str, float], float]:
    """Generate local batches and compute one globally gathered T2M summary."""

    if str(generation_mode) != "stream":
        raise ValueError("canonical batched T2M evaluation supports stream mode only")
    local_batches = tuple(batches)
    local_total = sum(len(batch.samples) for batch in local_batches)
    metric = T2MMetrics(metric_config).to(module.device).eval()
    started = time.perf_counter()
    completed = 0

    for batch in local_batches:
        frames = int(batch.frame_count)
        sample_indices = [int(item[0]) for item in batch.samples]
        samples = [item[1] for item in batch.samples]
        seeds = [
            stable_t2m_seed(
                base_seed,
                "t2m",
                sample["dataset"],
                sample["name"],
            )
            for sample in samples
        ]
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if module.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            generated = generate_t2m_evaluation_batch(
                module,
                samples,
                guidance_mode=str(guidance_mode),
                cfg_scale_joint=float(cfg_scale_joint),
                seeds=seeds,
                frame_count=frames,
                num_denoise_steps=int(num_denoise_steps),
            )
        with torch.inference_mode():
            target_root = torch.stack(
                [sample["root_motion"][:frames] for sample in samples],
                dim=0,
            ).to(module.device)
            target_body = torch.stack(
                [sample["body_motion"][:frames] for sample in samples],
                dim=0,
            ).to(module.device)
            reference = convert_root5_body259_to_humanml263(
                target_root,
                target_body,
                tail="drop",
            ).detach()
            predicted = convert_root5_body259_to_humanml263(
                generated.root_motion,
                generated.body_motion,
                tail="drop",
            ).detach()
            lengths = [int(reference.shape[1])] * len(samples)
            metric.update(
                reference,
                predicted,
                lengths,
                lengths,
                [list(prompt.tokens) for prompt in generated.prompts],
                sample_indices=sample_indices,
            )
        completed += len(samples)
        if progress_callback is not None:
            progress_callback(
                completed,
                local_total,
                len(samples),
                time.perf_counter() - started,
            )

    metric_seed = stable_t2m_seed(base_seed, "t2m_metric", generation_mode)
    torch.random.default_generator.manual_seed(metric_seed)
    np.random.seed(metric_seed % (2**32))
    values = metric.compute(bool(sanity_flag))
    summary = {
        key: float(value.detach().cpu().item())
        for key, value in values.items()
    }
    return summary, time.perf_counter() - started


__all__ = [
    "T2MEvaluationBatch",
    "build_t2m_evaluation_batches",
    "evaluate_t2m_batches",
    "shard_t2m_evaluation_batches",
    "stable_t2m_seed",
    "t2m_frame_count",
]
