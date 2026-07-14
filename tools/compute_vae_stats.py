"""Compute train-split root/local-root/body statistics for strict4 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.conditions.vae import BODY_CONTINUOUS_DIM, CONTRACT_VERSION
from utils.motion_representation import MotionStatistics, derive_patched_local_root


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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise RuntimeError("STRICT4_NATIVE_ROTATIONS_REQUIRED: strict4 manifest is missing")
    root_acc, local_acc, body_acc = Accumulator(5), Accumulator(4), Accumulator(BODY_CONTINUOUS_DIM)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    train = [record for record in records if record.get("split") == "train"]
    if not train:
        raise RuntimeError("statistics must be computed from a non-empty train split")
    for record in train:
        if record.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("manifest contract version mismatch")
        with np.load(manifest.parent / record["artifact"], allow_pickle=False) as data:
            root = torch.from_numpy(data["root_motion"]).float()[None]
            body = torch.from_numpy(data["body_motion"]).float()[None]
            body_valid = torch.from_numpy(data["body_feature_valid_mask"]).bool()[None]
        root_acc.update(root, torch.ones_like(root, dtype=torch.bool))
        local, local_valid = derive_patched_local_root(root, None, fps=args.fps)
        local_acc.update(local, local_valid)
        body_acc.update(body[..., :BODY_CONTINUOUS_DIM], body_valid[..., :BODY_CONTINUOUS_DIM])
    root_mean, root_std = root_acc.finish()
    local_mean, local_std = local_acc.finish()
    body_mean, body_std = body_acc.finish()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    MotionStatistics(
        root_mean, root_std, local_mean, local_std, body_mean, body_std,
        metadata={
            "contract_version": CONTRACT_VERSION,
            "split": "train",
            "fps": args.fps,
            "manifest_sha256": digest,
            "skeleton": "humanml22-native-rotations-v1",
        },
    ).save(args.output)
    print(f"wrote strict4 motion statistics to {args.output}")


if __name__ == "__main__":
    main()
