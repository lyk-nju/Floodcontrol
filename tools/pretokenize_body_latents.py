"""Encode strict4 body artifacts with deterministic posterior means."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    args = parser.parse_args()
    manifest = Path(args.manifest)
    checkpoint = Path(args.checkpoint)
    if not manifest.is_file() or not checkpoint.is_file():
        raise RuntimeError("strict4 manifest and frozen VAE checkpoint are required")
    model = load_model(args)
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    encoded = []
    train_values = []
    with torch.no_grad():
        for record in records:
            if record.get("contract_version") != CONTRACT_VERSION:
                raise ValueError("manifest contract version mismatch")
            with np.load(manifest.parent / record["artifact"], allow_pickle=False) as data:
                body = torch.from_numpy(data["body_motion"]).float()[None].to(args.device)
            valid = torch.ones(body.shape[:2], dtype=torch.bool, device=body.device)
            mu = model.encode(body, valid).mu[0].cpu()
            encoded.append(mu)
            if record.get("split") == "train":
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
    output_records = []
    for record, mu in zip(records, encoded, strict=True):
        path = artifact_root / f"{record['name']}.npy"
        np.save(path, ((mu - mean) / std).numpy())
        output_records.append({
            **record,
            "latent_artifact": str(path.relative_to(output)),
            "vae_checkpoint_sha256": ckpt_sha,
        })
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "latent_dim": args.latent_dim,
        "vae_checkpoint_sha256": ckpt_sha,
        "motion_stats_sha256": checkpoint_hash(Path(args.motion_stats)),
        "source_manifest_sha256": checkpoint_hash(manifest),
    }
    np.savez(
        output / "latent_stats.npz",
        latent_mu_mean=mean.numpy(),
        latent_mu_std=std.numpy(),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    with (output / "manifest.jsonl").open("w") as handle:
        for record in output_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {len(encoded)} normalized latent artifacts to {output}")


if __name__ == "__main__":
    main()
