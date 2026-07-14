"""Encode body motion artifacts with deterministic posterior means."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.humanml3d import load_humanml3d_records
from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import CONTRACT_VERSION


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(args) -> BodyVAE:
    model = BodyVAE(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        motion_stats_path=args.motion_stats,
        require_latent_statistics=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if "body_cont_mean" not in state and "model.body_cont_mean" in state:
        state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    model.load_state_dict(state, strict=True)
    return model.eval().to(args.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-meta-paths", nargs="+", required=True)
    parser.add_argument("--val-meta-paths", nargs="*", default=[])
    parser.add_argument("--test-meta-paths", nargs="*", default=[])
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError("a frozen VAE checkpoint is required")
    model = load_model(args)
    split_meta_paths = {
        "train": args.train_meta_paths,
        "val": args.val_meta_paths,
        "test": args.test_meta_paths,
    }
    split_records = {
        split: load_humanml3d_records(paths, artifact_path=args.artifact_path)
        for split, paths in split_meta_paths.items()
        if paths
    }
    encoded: list[tuple[str, dict[str, object], torch.Tensor]] = []
    train_values = []
    with torch.no_grad():
        for split, records in split_records.items():
            for record in records:
                with np.load(Path(record["artifact"]), allow_pickle=False) as data:
                    if (
                        "contract_version" not in data
                        or str(np.asarray(data["contract_version"]).item())
                        != CONTRACT_VERSION
                    ):
                        raise ValueError("motion artifact contract version mismatch")
                    body = (
                        torch.from_numpy(data["body_motion"])
                        .float()[None]
                        .to(args.device)
                    )
                valid = torch.ones(body.shape[:2], dtype=torch.bool, device=body.device)
                mu = model.encode(body, valid).mu[0].cpu()
                encoded.append((split, record, mu))
                if split == "train":
                    train_values.append(mu)
    if not train_values:
        raise RuntimeError("latent statistics require a non-empty train split")
    train = torch.cat(train_values, dim=0)
    mean = train.mean(0)
    std = train.std(0, unbiased=False).clamp_min(1e-6)
    output = Path(args.output)
    artifact_root = output / "latents"
    artifact_root.mkdir(parents=True, exist_ok=True)
    ckpt_sha = checkpoint_hash(checkpoint)
    output_names: dict[str, list[str]] = {split: [] for split in split_records}
    seen_names: set[str] = set()
    for split, record, mu in encoded:
        name = str(record["name"])
        if name in seen_names:
            raise ValueError(f"duplicate sample id across latent splits: {name!r}")
        seen_names.add(name)
        path = artifact_root / f"{name}.npy"
        np.save(path, ((mu - mean) / std).numpy())
        output_names[split].append(name)
    split_digest = hashlib.sha256()
    for split, paths in split_meta_paths.items():
        for meta_value in paths:
            meta_path = Path(meta_value)
            split_digest.update(split.encode())
            split_digest.update(str(meta_path.resolve()).encode())
            split_digest.update(meta_path.read_bytes())
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "latent_dim": args.latent_dim,
        "vae_checkpoint_sha256": ckpt_sha,
        "motion_stats_sha256": checkpoint_hash(Path(args.motion_stats)),
        "source_split_sha256": split_digest.hexdigest(),
    }
    np.savez(
        output / "latent_stats.npz",
        latent_mu_mean=mean.numpy(),
        latent_mu_std=std.numpy(),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    for split, names in output_names.items():
        (output / f"{split}.txt").write_text(
            "".join(f"{name}\n" for name in names)
        )
    print(f"wrote {len(encoded)} normalized latent artifacts to {output}")


if __name__ == "__main__":
    main()
