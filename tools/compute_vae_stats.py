"""Compute VAE statistics from full samples returned by a Dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from datasets.humanml3d import HumanML3DDataset
from utils.initialize import load_config
from utils.conditions.vae import BODY_CONTINUOUS_DIM
from utils.motion_process import recover_local_root
from utils.token_frame import MOTION_FPS
from utils.training.vae.data import create_dataset


class Accumulator:
    def __init__(self, dim: int):
        self.total = torch.zeros(dim, dtype=torch.float64)
        self.square = torch.zeros(dim, dtype=torch.float64)
        self.count = torch.zeros(dim, dtype=torch.float64)

    def update(self, value: torch.Tensor, valid: torch.Tensor) -> None:
        if not bool(torch.isfinite(value[valid]).all()):
            raise ValueError("statistics input contains non-finite valid values")
        value = value.reshape(-1, value.shape[-1]).double()
        valid = valid.reshape_as(value).double()
        self.total += (value * valid).sum(0)
        self.square += (value.square() * valid).sum(0)
        self.count += valid.sum(0)

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if bool((self.count == 0).any()):
            raise RuntimeError("statistics contain features without valid train samples")
        mean = self.total / self.count
        variance = self.square / self.count - mean.square()
        mean = mean.float()
        std = variance.clamp_min(1e-12).sqrt().float()
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("computed statistics contain non-finite values")
        if bool((std <= 0).any()):
            raise ValueError("computed statistics must have positive std")
        return mean, std


def compute_motion_statistics(
    dataset,
    *,
    fps: float = 20.0,
) -> dict[str, torch.Tensor]:
    """Accumulate yaw-invariant Body259 statistics from one Dataset."""

    local_acc = Accumulator(4)
    body_acc = Accumulator(BODY_CONTINUOUS_DIM)
    for index, sample in enumerate(dataset, start=1):
        root = sample["root_motion"][None]
        body = sample["body_motion"][None]
        body_valid = sample["body_feature_valid_mask"][None]
        body_acc.update(
            body[..., :BODY_CONTINUOUS_DIM],
            body_valid[..., :BODY_CONTINUOUS_DIM],
        )
        local, local_valid = recover_local_root(root, None, fps=fps)
        local_acc.update(local, local_valid)
        if index % 500 == 0 or index == len(dataset):
            print(f"processed statistics for {index}/{len(dataset)} samples", flush=True)
    local_mean, local_std = local_acc.finish()
    body_mean, body_std = body_acc.finish()
    return {
        "local_root_mean": local_mean,
        "local_root_std": local_std,
        "body_cont_mean": body_mean,
        "body_cont_std": body_std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--override", nargs="+")
    parser.add_argument("--train-meta-paths", nargs="+")
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--output")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    if args.config:
        overrides = {}
        for item in args.override or ():
            key, separator, value = item.partition("=")
            if separator != "=" or not key.strip():
                parser.error(f"invalid --override {item!r}; expected KEY=VALUE")
            overrides[key.strip()] = value.strip()
        cfg = load_config(args.config, overrides)
        dataset = create_dataset(cfg, "train")
        fps = MOTION_FPS
        output_path = args.output or str(cfg.model.params.motion_stats_path)
    else:
        if args.override:
            parser.error("--override requires --config")
        if not args.train_meta_paths:
            parser.error("set --config or --train-meta-paths")
        if not args.output:
            parser.error("--output is required without --config")
        dataset = HumanML3DDataset(
            meta_paths=args.train_meta_paths,
            split="train",
            artifact_path=args.artifact_path,
            fps=args.fps,
        )
        fps = float(args.fps)
        output_path = args.output
    statistics = compute_motion_statistics(dataset, fps=fps)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        **{name: value.numpy() for name, value in statistics.items()},
    )
    print(f"wrote motion statistics to {output}")


if __name__ == "__main__":
    main()
