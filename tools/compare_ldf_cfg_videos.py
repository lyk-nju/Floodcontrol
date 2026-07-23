#!/usr/bin/env python3
"""Render a fixed-noise dense-XZ video comparison across joint CFG scales."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from metrics.trajectory import compute_dense_xz_metrics, compute_foot_skating_ratio
from utils.initialize import ProjectConfig
from utils.training.ldf.data import create_dataset
from utils.training.ldf.evaluation.artifacts import (
    render_comparison_video,
    write_json,
)
from utils.training.ldf.evaluation.generation import generate_evaluation_sequence
from utils.training.ldf.lightning_module import LDFLightningModule
from utils.training.ldf.metrics import compute_heading_metrics


@dataclass(frozen=True)
class CFGVariant:
    """One inference branch in the comparison."""

    name: str
    mode: str
    joint_scale: float


def comparison_variants(
    joint_scales: Iterable[float] = (1.0, 2.0, 3.0),
) -> tuple[CFGVariant, ...]:
    """Return no-CFG followed by the requested joint-CFG scales."""

    variants = [CFGVariant(name="nocfg", mode="nocfg", joint_scale=1.0)]
    seen_names = {"nocfg"}
    for raw_scale in joint_scales:
        scale = float(raw_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("joint CFG scales must be finite and non-negative")
        label = f"joint_{scale:.2f}".replace(".", "p")
        if label in seen_names:
            raise ValueError(f"duplicate joint CFG scale {scale:g}")
        seen_names.add(label)
        variants.append(CFGVariant(name=label, mode="joint", joint_scale=scale))
    return tuple(variants)


def _parse_override_value(value: str):
    lowered = value.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _load_config(path: str, items: list[str]):
    overrides: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must use KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override key must not be empty: {item!r}")
        overrides[key] = _parse_override_value(value.strip())
    return ProjectConfig(path, overrides=overrides).config


def _load_module(
    cfg,
    checkpoint_path: str | Path,
    device: torch.device,
) -> LDFLightningModule:
    module = LDFLightningModule(cfg)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    module.on_load_checkpoint(checkpoint)
    module.to(device)
    module.eval()
    module.vae.eval()
    module.ema.to(device)
    return module


def _generation_dataset(cfg, split: str, probe: str | None):
    if probe is None:
        return create_dataset(cfg, split)
    paths = cfg.data.get("test_probe_meta_paths") or {}
    if probe not in paths:
        raise KeyError(f"data.test_probe_meta_paths has no probe {probe!r}")
    return create_dataset(cfg, "test", meta_paths=list(paths[probe]))


def _select_samples(dataset, sample_ids: tuple[str, ...]) -> list[dict[str, object]]:
    requested = set(sample_ids)
    selected: dict[str, dict[str, object]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        name = str(sample["name"])
        if name in requested:
            selected[name] = sample
            if len(selected) == len(requested):
                break
    missing = [name for name in sample_ids if name not in selected]
    if missing:
        raise RuntimeError(f"samples were not found: {', '.join(missing)}")
    return [selected[name] for name in sample_ids]


def _resolve_sample_ids(cfg, values: list[str] | None) -> tuple[str, ...]:
    if values:
        return tuple(str(value) for value in values)
    dense = cfg.validation.get("dense_xz") or {}
    standard = dense.get("standard_cases") or ()
    if not standard:
        raise ValueError(
            "provide --sample-ids or set validation.dense_xz.standard_cases"
        )
    return tuple(str(value) for value in standard)


def _resolve_frames(sample: dict[str, object], maximum: int, requested: int) -> int:
    frames = min(int(sample["root_motion"].shape[0]), int(maximum))
    if requested > 0:
        frames = min(frames, int(requested))
    frames -= frames % 4
    if frames <= 0:
        raise ValueError(f"sample {sample['name']!r} has no complete four-frame token")
    return frames


def _metrics(
    generated,
    *,
    target_root: torch.Tensor,
    target_body: torch.Tensor,
    fps: float,
) -> dict[str, float | int]:
    record: dict[str, float | int] = compute_dense_xz_metrics(
        generated.root_motion,
        target_root,
    )
    record["foot_skating_ratio"] = compute_foot_skating_ratio(
        generated.root_motion,
        generated.body_motion,
        fps=fps,
    )
    frame_mask = torch.ones(
        1,
        target_root.shape[0],
        device=generated.root_motion.device,
        dtype=torch.bool,
    )
    heading = compute_heading_metrics(
        predicted_root=generated.root_motion[None],
        target_root=target_root.to(generated.root_motion)[None],
        predicted_body=generated.body_motion[None],
        target_body=target_body.to(generated.body_motion)[None],
        frame_mask=frame_mask,
        fps=fps,
    )
    record.update(
        {
            name: float(value.detach().cpu())
            for name, value in heading.items()
        }
    )
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ldf.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="Defaults to validation.dense_xz.standard_cases",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Defaults to validation.seed plus validation.generation.num_runs",
    )
    parser.add_argument(
        "--joint-scales",
        nargs="+",
        type=float,
        default=(1.0, 2.0, 3.0),
    )
    parser.add_argument("--probe", default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--generation-mode",
        choices=("stream", "rolling"),
        default="stream",
    )
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--max-horizon-token", type=int)
    parser.add_argument("--num-denoise-steps", type=int)
    parser.add_argument("--rolling-window-tokens", type=int)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output", default="outputs/cfg_video_comparison")
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Use raw training weights instead of EMA weights",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_config(args.config, args.override)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    generation_cfg = cfg.validation.generation
    dense_cfg = cfg.validation.get("dense_xz") or {}
    probe = args.probe
    if probe is None and args.split == "test":
        probe = dense_cfg.get("probe")
    sample_ids = _resolve_sample_ids(cfg, args.sample_ids)
    dataset = _generation_dataset(cfg, args.split, probe)
    samples = _select_samples(dataset, sample_ids)
    if args.seeds:
        seeds = tuple(int(value) for value in args.seeds)
    else:
        base_seed = int(cfg.validation.get("seed", cfg.get("seed", 0)))
        run_count = int(generation_cfg.get("num_runs", 1))
        seeds = tuple(base_seed + index for index in range(run_count))
    variants = comparison_variants(args.joint_scales)

    max_horizon = (
        int(generation_cfg.max_horizon_token)
        if args.max_horizon_token is None
        else int(args.max_horizon_token)
    )
    denoise_steps = (
        int(generation_cfg.num_denoise_steps)
        if args.num_denoise_steps is None
        else int(args.num_denoise_steps)
    )
    rolling_tokens = (
        int(generation_cfg.rolling.window_tokens)
        if args.rolling_window_tokens is None
        else int(args.rolling_window_tokens)
    )
    module = _load_module(cfg, args.checkpoint, device)
    checkpoint_name = Path(args.checkpoint).stem
    output_root = Path(args.output) / checkpoint_name
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "weights": "raw" if args.raw_weights else "ema",
        "generation_mode": args.generation_mode,
        "dense_xz": True,
        "max_horizon_token": max_horizon,
        "num_denoise_steps": denoise_steps,
        "variants": [
            {
                "name": variant.name,
                "mode": variant.mode,
                "joint_scale": variant.joint_scale,
            }
            for variant in variants
        ],
        "runs": [],
    }

    parameter_scope = nullcontext() if args.raw_weights else module.use_ema_parameters()
    with parameter_scope:
        for sample in samples:
            frames = _resolve_frames(sample, int(cfg.data.max_frames), args.frames)
            target_root = sample["root_motion"][:frames].cpu().float()
            target_body = sample["body_motion"][:frames].cpu().float()
            for seed in seeds:
                for variant in variants:
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with torch.inference_mode(), autocast:
                        generated = generate_evaluation_sequence(
                            module,
                            sample,
                            mode=args.generation_mode,
                            guidance_mode=variant.mode,
                            cfg_scale_joint=variant.joint_scale,
                            seed=seed,
                            frame_count=frames,
                            dense_xz=True,
                            rolling_window_tokens=rolling_tokens,
                            max_horizon_token=max_horizon,
                            num_denoise_steps=denoise_steps,
                        )
                    run_dir = (
                        output_root
                        / str(sample["name"])
                        / f"seed_{seed}"
                        / variant.name
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)
                    generated_root = generated.root_motion.detach().cpu().float()
                    generated_body = generated.body_motion.detach().cpu().float()
                    record = _metrics(
                        generated,
                        target_root=target_root,
                        target_body=target_body,
                        fps=float(module.model.fps),
                    )
                    record.update(
                        {
                            "dataset": str(sample["dataset"]),
                            "sample_id": str(sample["name"]),
                            "seed": int(seed),
                            "cfg_mode": variant.mode,
                            "cfg_scale_joint": float(variant.joint_scale),
                            "frames": int(frames),
                        }
                    )
                    np.savez(
                        run_dir / "motion.npz",
                        root_motion=generated_root.numpy(),
                        body_motion=generated_body.numpy(),
                        target_root_motion=target_root.numpy(),
                        target_body_motion=target_body.numpy(),
                    )
                    write_json(run_dir / "metrics.json", record)
                    render_comparison_video(
                        target_root=target_root,
                        target_body=target_body,
                        predicted_root=generated_root,
                        predicted_body=generated_body,
                        predicted_video_path=run_dir / "generated.mp4",
                        composite_path=run_dir / "comparison.mp4",
                        caption=(
                            f"{generated.prompt.caption} | {variant.mode} "
                            f"scale={variant.joint_scale:g} | seed={seed}"
                        ),
                        fps=float(module.model.fps),
                    )
                    summary["runs"].append(record)
                    print(
                        f"{sample['name']} seed={seed} {variant.name}: "
                        f"ADE={record['ade']:.4f} m, "
                        f"root/GT={record['root_gt_root_heading_angle_deg']:.2f} deg"
                    )

    write_json(output_root / "summary.json", summary)
    print(f"wrote CFG video comparison to {output_root}")


if __name__ == "__main__":
    main()
