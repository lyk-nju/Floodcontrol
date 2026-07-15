"""Compute seeded posterior-mu statistics from full Dataset samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.humanml3d import HumanML3DDataset
from models.vae_wan_1d import BodyVAE
from utils.initialize import load_config
from utils.motion_process import rotate_motion_yaw
from utils.token_frame import frame_count_to_token_count
from utils.training.vae.checkpoint import load_vae_checkpoint


class LatentStatisticsAccumulator:
    def __init__(self, dim: int):
        self.total = torch.zeros(dim, dtype=torch.float64)
        self.square = torch.zeros(dim, dtype=torch.float64)
        self.count = 0

    def update(self, value: torch.Tensor, *, sample_identity: str) -> None:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(
                f"non-finite posterior mu for {sample_identity}"
            )
        flattened = value.detach().reshape(-1, value.shape[-1]).double().cpu()
        self.total += flattened.sum(0)
        self.square += flattened.square().sum(0)
        self.count += int(flattened.shape[0])

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise RuntimeError("latent statistics require a non-empty train split")
        mean = (self.total / self.count).float()
        variance = self.square / self.count - mean.double().square()
        std = variance.clamp_min(1e-12).sqrt().float()
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("computed latent statistics contain non-finite values")
        if bool((std <= 0).any()):
            raise ValueError("computed latent standard deviation must be positive")
        return mean, std


def _load_model(
    config_path: str,
    checkpoint_path: Path,
    motion_stats_path: str | Path,
    device: torch.device,
) -> BodyVAE:
    cfg = load_config(config_path)
    params = OmegaConf.to_container(cfg.model.params, resolve=True)
    params["motion_stats_path"] = str(motion_stats_path)
    params["latent_stats_path"] = None
    model = BodyVAE(**params)
    load_vae_checkpoint(model, checkpoint_path)
    return model.to(device)


@torch.no_grad()
def compute_latent_statistics(
    model: BodyVAE,
    dataset,
    *,
    device: torch.device,
    yaw_seed: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    accumulator = LatentStatisticsAccumulator(model.latent_dim)
    yaw_generator = torch.Generator(device="cpu")
    yaw_generator.manual_seed(int(yaw_seed))
    for start in range(0, len(dataset), batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        max_frames = max(int(sample["body_motion"].shape[0]) for sample in samples)
        roots = torch.zeros(len(samples), max_frames, 5)
        roots[..., 3] = 1.0
        bodies = torch.zeros(len(samples), max_frames, 265)
        frame_valid = torch.zeros(len(samples), max_frames, dtype=torch.bool)
        lengths: list[int] = []
        for index, sample in enumerate(samples):
            root, body = sample["root_motion"], sample["body_motion"]
            frames = int(body.shape[0])
            roots[index, :frames] = root
            bodies[index, :frames] = body
            frame_valid[index, :frames] = True
            lengths.append(frame_count_to_token_count(frames))
        angles = torch.rand(len(samples), generator=yaw_generator) * (2.0 * torch.pi)
        _, bodies = rotate_motion_yaw(
            roots.to(device),
            bodies.to(device),
            angles.to(device),
        )
        posterior = model.encode(bodies, frame_valid.to(device))
        for index, sample in enumerate(samples):
            accumulator.update(
                posterior.mu[index, : lengths[index]],
                sample_identity=f"{sample['dataset']}/{sample['name']}",
            )
        completed = min(start + batch_size, len(dataset))
        if completed % 512 == 0 or completed == len(dataset):
            print(
                f"processed latent statistics for {completed}/{len(dataset)} samples",
                flush=True,
            )
    mean, std = accumulator.finish()
    return mean, std, accumulator.count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-meta-paths", nargs="+", required=True)
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--config", default="configs/vae.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--yaw-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise RuntimeError("VAE training checkpoint is required")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(args.device)
    model = _load_model(
        args.config, checkpoint_path, args.motion_stats, device
    )
    dataset = HumanML3DDataset(
        meta_paths=args.train_meta_paths,
        split="train",
        artifact_path=args.artifact_path,
        fps=model.fps,
    )
    mean, std, token_count = compute_latent_statistics(
        model,
        dataset,
        device=device,
        yaw_seed=args.yaw_seed,
        batch_size=args.batch_size,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.npz")
    try:
        np.savez(temporary, mean=mean.numpy(), std=std.numpy())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote latent statistics to {output}")


if __name__ == "__main__":
    main()
