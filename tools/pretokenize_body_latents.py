"""Optional diagnostic/export tool for namespaced EMA body latents.

Formal LDF training encodes deterministic posterior ``mu`` online and does not
consume the artifacts written here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from datasets.humanml3d import load_humanml3d_records
from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import CONTRACT_VERSION, FRAMES_PER_TOKEN
from utils.motion_representation import (
    MOTION_CONVERTER_VERSION,
    deterministic_sample_yaw,
    motion_artifact_manifest_sha256,
    rotate_root_body_yaw,
)
from utils.training.vae.checkpoint import (
    TOKENIZER_FORMAT_VERSION,
    load_tokenizer_bundle,
    sha256_file,
)


YAW_POLICY = "sample-id-sha256-uniform-v1"


class ChannelAccumulator:
    def __init__(self, dim: int):
        self.total = torch.zeros(dim, dtype=torch.float64)
        self.square = torch.zeros(dim, dtype=torch.float64)
        self.count = 0

    def update(self, value: torch.Tensor) -> None:
        value = value.reshape(-1, value.shape[-1]).double()
        self.total += value.sum(0)
        self.square += value.square().sum(0)
        self.count += int(value.shape[0])

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise RuntimeError("latent statistics require a non-empty train split")
        mean = self.total / self.count
        variance = self.square / self.count - mean.square()
        return mean.float(), variance.clamp_min(1e-12).sqrt().float()


def _load_model(tokenizer_path: Path, device: str) -> tuple[BodyVAE, dict[str, object]]:
    bundle = torch.load(tokenizer_path, map_location="cpu", weights_only=False, mmap=True)
    if bundle.get("format_version") != TOKENIZER_FORMAT_VERSION:
        raise ValueError("formal EMA tokenizer bundle is required")
    model_config = bundle.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError("tokenizer bundle model_config is missing")
    model = BodyVAE(
        **model_config,
        allow_identity_statistics=True,
        require_latent_statistics=False,
    )
    metadata = load_tokenizer_bundle(model, tokenizer_path)
    return model.eval().to(device), metadata


def _read_motion(record: dict[str, object], *, expected_fps: float) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(record["artifact"])
    with np.load(path, allow_pickle=False) as data:
        contract = str(np.asarray(data["contract_version"]).item())
        converter = str(np.asarray(data["converter_version"]).item())
        fps = float(np.asarray(data["fps"]).item())
        root = torch.from_numpy(data["root_motion"]).float()
        body = torch.from_numpy(data["body_motion"]).float()
    if contract != CONTRACT_VERSION:
        raise ValueError(f"motion artifact contract version mismatch in {path}")
    if converter != MOTION_CONVERTER_VERSION:
        raise ValueError(f"motion artifact converter version mismatch in {path}")
    if not np.isclose(fps, expected_fps, rtol=0.0, atol=1e-6):
        raise ValueError(f"motion artifact FPS mismatch in {path}")
    if root.shape[0] != body.shape[0] or root.shape[0] % FRAMES_PER_TOKEN:
        raise ValueError(f"motion artifact frame contract mismatch in {path}")
    return root, body


def _encoded_batches(
    model: BodyVAE,
    records: list[dict[str, object]],
    *,
    batch_size: int,
    device: str,
    yaw_seed: int,
):
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            motions = [
                _read_motion(record, expected_fps=model.fps)
                for record in batch_records
            ]
            max_frames = max(body.shape[0] for _, body in motions)
            roots = torch.zeros(len(motions), max_frames, 5, dtype=torch.float32)
            bodies = torch.zeros(len(motions), max_frames, 265, dtype=torch.float32)
            valid = torch.zeros(len(motions), max_frames, dtype=torch.bool)
            angles = []
            frame_lengths = []
            for index, (record, (root, body)) in enumerate(zip(batch_records, motions, strict=True)):
                frames = int(body.shape[0])
                roots[index, :frames] = root
                bodies[index, :frames] = body
                valid[index, :frames] = True
                frame_lengths.append(frames)
                angles.append(
                    deterministic_sample_yaw(
                        str(record["dataset"]), str(record["name"]), seed=yaw_seed
                    )
                )
            roots, bodies = rotate_root_body_yaw(
                roots.to(device),
                bodies.to(device),
                torch.tensor(angles, device=device),
            )
            posterior = model.encode(bodies, valid.to(device))
            for index, record in enumerate(batch_records):
                tokens = frame_lengths[index] // FRAMES_PER_TOKEN
                yield record, angles[index], posterior.mu[index, :tokens].cpu()


def _atomic_save_latent(
    path: Path,
    normalized_mu: torch.Tensor,
    *,
    yaw_offset: float,
    dataset: str,
    sample_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            latent_motion=normalized_mu.numpy(),
            yaw_offset=np.float32(yaw_offset),
            dataset=dataset,
            sample_id=sample_id,
            contract_version=CONTRACT_VERSION,
            converter_version=MOTION_CONVERTER_VERSION,
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-meta-paths", nargs="+", required=True)
    parser.add_argument("--val-meta-paths", nargs="*", default=[])
    parser.add_argument("--test-meta-paths", nargs="*", default=[])
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--yaw-seed", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        raise RuntimeError("a formal EMA tokenizer bundle is required")
    model, tokenizer_metadata = _load_model(tokenizer_path, args.device)
    if tokenizer_metadata.get("weights_kind") != "ema":
        raise ValueError("latent artifacts require EMA encoder weights")

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

    accumulator = ChannelAccumulator(model.latent_dim)
    for _, _, mu in _encoded_batches(
        model,
        split_records["train"],
        batch_size=args.batch_size,
        device=args.device,
        yaw_seed=args.yaw_seed,
    ):
        accumulator.update(mu)
    mean, std = accumulator.finish()

    output = Path(args.output)
    artifact_root = output / "latents"
    output_names: dict[str, list[str]] = {split: [] for split in split_records}
    seen: set[tuple[str, str]] = set()
    for split, records in split_records.items():
        for record, yaw_offset, mu in _encoded_batches(
            model,
            records,
            batch_size=args.batch_size,
            device=args.device,
            yaw_seed=args.yaw_seed,
        ):
            dataset, name = str(record["dataset"]), str(record["name"])
            identity = (dataset, name)
            if identity in seen:
                raise ValueError(f"duplicate sample across latent splits: {dataset}/{name}")
            seen.add(identity)
            relative = Path(dataset) / f"{name}.npz"
            _atomic_save_latent(
                artifact_root / relative,
                (mu - mean) / std,
                yaw_offset=yaw_offset,
                dataset=dataset,
                sample_id=name,
            )
            output_names[split].append(str(relative.with_suffix("")))

    split_digest = hashlib.sha256()
    for split, paths in split_meta_paths.items():
        for meta_value in paths:
            meta_path = Path(meta_value)
            split_digest.update(split.encode("utf-8"))
            split_digest.update(meta_path.parent.name.encode("utf-8"))
            split_digest.update(meta_path.read_bytes())
    artifact_manifests = {
        split: motion_artifact_manifest_sha256(
            records, expected_fps=model.fps
        )[0]
        for split, records in split_records.items()
    }
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "converter_version": MOTION_CONVERTER_VERSION,
        "latent_dim": model.latent_dim,
        "weights_kind": "ema",
        "tokenizer_format_version": TOKENIZER_FORMAT_VERSION,
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "training_checkpoint_sha256": tokenizer_metadata["training_checkpoint_sha256"],
        "inference_state_sha256": tokenizer_metadata["inference_state_sha256"],
        "motion_stats_sha256": tokenizer_metadata["motion_stats_sha256"],
        "source_split_sha256": split_digest.hexdigest(),
        "source_artifact_manifest_sha256": artifact_manifests,
        "yaw_policy": YAW_POLICY,
        "yaw_seed": args.yaw_seed,
        "train_token_count": accumulator.count,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / "latent_stats.npz",
        latent_mu_mean=mean.numpy(),
        latent_mu_std=std.numpy(),
        metadata=json.dumps(metadata, sort_keys=True),
    )
    for split, names in output_names.items():
        (output / f"{split}.txt").write_text("".join(f"{name}\n" for name in names))
    print(f"wrote {len(seen)} normalized EMA latent artifacts to {output}")


if __name__ == "__main__":
    main()
