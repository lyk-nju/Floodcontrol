"""Compute train-split root/local-root/body statistics for motion artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.humanml3d import load_humanml3d_records
from utils.conditions.vae import BODY_CONTINUOUS_DIM, CONTRACT_VERSION
from utils.motion_representation import (
    MotionStatistics,
    derive_patched_local_root,
    rotate_root_body_yaw,
)


class Accumulator:
    def __init__(self, dim: int):
        self.total = torch.zeros(dim, dtype=torch.float64)
        self.square = torch.zeros(dim, dtype=torch.float64)
        self.count = torch.zeros(dim, dtype=torch.float64)

    def update(self, value: torch.Tensor, valid: torch.Tensor) -> None:
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
        return mean.float(), variance.clamp_min(1e-12).sqrt().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-meta-paths", nargs="+", required=True)
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    root_acc = Accumulator(5)
    local_acc = Accumulator(4)
    body_acc = Accumulator(BODY_CONTINUOUS_DIM)
    records = load_humanml3d_records(
        args.train_meta_paths, artifact_path=args.artifact_path
    )
    source_representations: set[str] = set()
    yaw_quadrature = torch.arange(4, dtype=torch.float32) * (torch.pi / 2)
    for record in records:
        with np.load(Path(record["artifact"]), allow_pickle=False) as data:
            if (
                "contract_version" not in data
                or str(np.asarray(data["contract_version"]).item())
                != CONTRACT_VERSION
            ):
                raise ValueError("motion artifact contract version mismatch")
            root = torch.from_numpy(data["root_motion"]).float()[None]
            body = torch.from_numpy(data["body_motion"]).float()[None]
            body_valid = torch.from_numpy(data["body_feature_valid_mask"]).bool()[None]
            source_representations.add(
                str(np.asarray(data["source_representation"]).item())
                if "source_representation" in data else "unknown"
            )
        # Training samples receive a fresh uniform global yaw. Four quarter-turn
        # quadrature points exactly match the first and per-feature second moments
        # of every x/z vector, heading pair, and global rotation column under a
        # continuous uniform yaw, without making statistics nondeterministic.
        for angle in yaw_quadrature:
            rotated_root, rotated_body = rotate_root_body_yaw(
                root, body, angle.reshape(1)
            )
            root_acc.update(
                rotated_root, torch.ones_like(rotated_root, dtype=torch.bool)
            )
            body_acc.update(
                rotated_body[..., :BODY_CONTINUOUS_DIM],
                body_valid[..., :BODY_CONTINUOUS_DIM],
            )
        local, local_valid = derive_patched_local_root(root, None, fps=args.fps)
        local_acc.update(local, local_valid)
    root_mean, root_std = root_acc.finish()
    local_mean, local_std = local_acc.finish()
    body_mean, body_std = body_acc.finish()
    digest = hashlib.sha256()
    for meta_value in args.train_meta_paths:
        meta_path = Path(meta_value)
        digest.update(str(meta_path.resolve()).encode())
        digest.update(meta_path.read_bytes())
    MotionStatistics(
        root_mean, root_std, local_mean, local_std, body_mean, body_std,
        metadata={
            "contract_version": CONTRACT_VERSION,
            "split": "train",
            "fps": args.fps,
            "train_meta_sha256": digest.hexdigest(),
            "train_meta_paths": [str(Path(path)) for path in args.train_meta_paths],
            "skeleton": "humanml22-v1",
            "source_representations": sorted(source_representations),
            "yaw_statistics": "uniform-four-point-quadrature-v1",
        },
    ).save(args.output)
    print(f"wrote motion statistics to {args.output}")


if __name__ == "__main__":
    main()
