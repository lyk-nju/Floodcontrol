"""Compute root5 statistics with the scaled-ARDY LDF window distribution."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from datasets.humanml3d import HumanML3DDataset
from utils.initialize import load_config
from utils.motion_process import ROOT_DIM, rotate_root_yaw
from utils.token_frame import FRAMES_PER_TOKEN
from utils.training.ldf.data import create_dataset


def root_statistics_recipe(cfg) -> dict[str, object]:
    """Resolve the offline root-statistics recipe from the current LDF config.

    ``root_statistics`` remains an optional tool-specific override.  When the
    formal training config omits it, the recipe follows the same window,
    generation band, and yaw policy already declared by the data/training
    contract instead of requiring duplicate configuration.
    """

    statistics = cfg.get("root_statistics") or {}
    training = cfg.get("training") or {}
    window = training.get("window") or {}
    window_tokens = int(
        statistics.get(
            "window_tokens",
            window.get("max_tokens", int(cfg.data.max_frames) // FRAMES_PER_TOKEN),
        )
    )
    generation_tokens = int(
        statistics.get(
            "generation_tokens",
            window.get("generation_tokens", cfg.model.params.chunk_size),
        )
    )
    recipe = {
        "window_tokens": window_tokens,
        "generation_tokens": generation_tokens,
        "anchor_sampling": str(
            statistics.get("anchor_sampling", "uniform_legal_history")
        ),
        "random_yaw": bool(
            statistics.get("random_yaw", cfg.data.get("random_yaw", True))
        ),
        "windows_per_sample": int(statistics.get("windows_per_sample", 1)),
    }
    if recipe["anchor_sampling"] != "uniform_legal_history":
        raise ValueError(
            "root statistics only support uniform_legal_history anchors"
        )
    return recipe


class RootAccumulator:
    def __init__(self):
        self.total = torch.zeros(ROOT_DIM, dtype=torch.float64)
        self.square = torch.zeros(ROOT_DIM, dtype=torch.float64)
        self.count = 0

    def update(self, root_motion: torch.Tensor) -> None:
        if root_motion.ndim != 2 or root_motion.shape[-1] != ROOT_DIM:
            raise ValueError("root_motion must be [F,5]")
        if not bool(torch.isfinite(root_motion).all()):
            raise ValueError("root statistics input contains non-finite values")
        values = root_motion.double()
        self.total += values.sum(dim=0)
        self.square += values.square().sum(dim=0)
        self.count += int(values.shape[0])

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise RuntimeError("root statistics received no frames")
        mean = self.total / self.count
        variance = self.square / self.count - mean.square()
        mean = mean.float()
        std = variance.clamp_min(1e-12).sqrt().float()
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("computed root statistics contain non-finite values")
        if bool((std <= 0).any()):
            raise ValueError("computed root statistics must have positive std")
        return mean, std


def compute_root_statistics(
    dataset,
    *,
    min_frames: int,
    max_frames: int,
    windows_per_sample: int = 1,
    random_yaw: bool = True,
    active_tokens: int = 5,
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample parent windows and a legal true-cold/continuation H per item."""

    for name, value in (("min_frames", min_frames), ("max_frames", max_frames)):
        if value < FRAMES_PER_TOKEN or value % FRAMES_PER_TOKEN:
            raise ValueError(f"{name} must be a positive multiple of four")
    if min_frames > max_frames:
        raise ValueError("min_frames must not exceed max_frames")
    if windows_per_sample <= 0:
        raise ValueError("windows_per_sample must be positive")
    if active_tokens <= 0:
        raise ValueError("active_tokens must be positive")
    if active_tokens > max_frames // FRAMES_PER_TOKEN:
        raise ValueError("active_tokens must fit inside max_frames")

    sampler = random.Random(int(seed))
    yaw_generator = torch.Generator().manual_seed(int(seed))
    accumulator = RootAccumulator()
    for index, sample in enumerate(dataset, start=1):
        full_root = sample["root_motion"]
        available = int(full_root.shape[0])
        if available < min_frames:
            continue
        available_tokens = available // FRAMES_PER_TOKEN
        parent_tokens = min(available_tokens, max_frames // FRAMES_PER_TOKEN)
        if parent_tokens < active_tokens:
            continue
        for _ in range(windows_per_sample):
            source_start = sampler.randint(
                0, available_tokens - parent_tokens
            )
            minimum_history = 0 if source_start == 0 else 1
            history_tokens = sampler.randint(
                minimum_history,
                parent_tokens - active_tokens,
            )
            start = source_start * FRAMES_PER_TOKEN
            frames = parent_tokens * FRAMES_PER_TOKEN
            root = full_root[start : start + frames].clone()
            anchor_frame = (
                history_tokens * FRAMES_PER_TOKEN - 1
                if history_tokens
                else 0
            )
            root[:, 0] -= root[anchor_frame, 0].clone()
            root[:, 2] -= root[anchor_frame, 2].clone()
            if random_yaw:
                angle = torch.rand(1, generator=yaw_generator) * (2.0 * torch.pi)
                root = rotate_root_yaw(root[None], angle)[0]
            accumulator.update(root)
        if index % 500 == 0 or index == len(dataset):
            print(f"processed root statistics for {index}/{len(dataset)} samples", flush=True)
    return accumulator.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--train-meta-paths", nargs="+")
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--output")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--windows-per-sample", type=int)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--random-yaw",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    if args.config:
        cfg = load_config(args.config)
        dataset = create_dataset(cfg, "train")
        statistics = root_statistics_recipe(cfg)
        min_frames = int(
            cfg.data.min_frames if args.min_frames is None else args.min_frames
        )
        max_frames = int(
            int(statistics["window_tokens"]) * FRAMES_PER_TOKEN
            if args.max_frames is None
            else args.max_frames
        )
        active_tokens = int(
            statistics["generation_tokens"]
            if args.active_tokens is None
            else args.active_tokens
        )
        windows_per_sample = int(
            statistics["windows_per_sample"]
            if args.windows_per_sample is None
            else args.windows_per_sample
        )
        random_yaw = bool(
            statistics["random_yaw"]
            if args.random_yaw is None
            else args.random_yaw
        )
        output_path = args.output or str(cfg.data.root_stats_path)
    else:
        if not args.train_meta_paths:
            parser.error("set --config or --train-meta-paths")
        if not args.output:
            parser.error("--output is required without --config")
        dataset = HumanML3DDataset(
            meta_paths=args.train_meta_paths,
            split="train",
            artifact_path=args.artifact_path,
            text_path=None,
            fps=args.fps,
        )
        min_frames = 20 if args.min_frames is None else args.min_frames
        max_frames = 200 if args.max_frames is None else args.max_frames
        active_tokens = 5 if args.active_tokens is None else args.active_tokens
        windows_per_sample = (
            1 if args.windows_per_sample is None else args.windows_per_sample
        )
        random_yaw = True if args.random_yaw is None else args.random_yaw
        output_path = args.output
    root_mean, root_std = compute_root_statistics(
        dataset,
        min_frames=min_frames,
        max_frames=max_frames,
        windows_per_sample=windows_per_sample,
        random_yaw=random_yaw,
        active_tokens=active_tokens,
        seed=args.seed,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "window_tokens": max_frames // FRAMES_PER_TOKEN,
        "generation_tokens": active_tokens,
        "anchor_sampling": "uniform_legal_history",
        "random_yaw": random_yaw,
        "windows_per_sample": windows_per_sample,
    }
    np.savez(
        output,
        root_mean=root_mean.numpy(),
        root_std=root_std.numpy(),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(f"wrote LDF root statistics to {output}")


if __name__ == "__main__":
    main()
