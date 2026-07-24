#!/usr/bin/env python3
"""Sweep no-CFG and joint-CFG scales with the canonical HumanML T2M metrics."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from utils.initialize import ProjectConfig
from utils.training.ldf.data import create_dataset
from utils.training.ldf.evaluation.artifacts import write_json
from utils.training.ldf.evaluation.t2m import (
    build_t2m_evaluation_batches,
    evaluate_t2m_batches,
)
from utils.training.ldf.lightning_module import LDFLightningModule


@dataclass(frozen=True)
class GuidanceVariant:
    """One guidance configuration in a paired T2M sweep."""

    name: str
    mode: str
    joint_scale: float


def guidance_variants(
    joint_scales: Iterable[float],
) -> tuple[GuidanceVariant, ...]:
    """Return no-CFG followed by unique, finite joint-CFG scales."""

    variants = [GuidanceVariant(name="nocfg", mode="nocfg", joint_scale=1.0)]
    names = {"nocfg"}
    for raw_scale in joint_scales:
        scale = float(raw_scale)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("joint CFG scales must be finite and non-negative")
        name = f"joint_{scale:.2f}".replace(".", "p")
        if name in names:
            raise ValueError(f"duplicate joint CFG scale {scale:g}")
        names.add(name)
        variants.append(
            GuidanceVariant(name=name, mode="joint", joint_scale=scale)
        )
    return tuple(variants)


def assign_variant_groups(
    variants: Iterable[GuidanceVariant],
    devices: Iterable[int],
) -> tuple[tuple[int, tuple[GuidanceVariant, ...]], ...]:
    """Assign complete CFG variants round-robin to independent GPU workers."""

    device_ids = tuple(int(device) for device in devices)
    if not device_ids:
        raise ValueError("at least one GPU device is required")
    if len(set(device_ids)) != len(device_ids) or any(
        device < 0 for device in device_ids
    ):
        raise ValueError("GPU device indices must be unique and non-negative")
    groups: list[list[GuidanceVariant]] = [[] for _ in device_ids]
    for index, variant in enumerate(variants):
        groups[index % len(device_ids)].append(variant)
    return tuple(
        (device, tuple(group))
        for device, group in zip(device_ids, groups)
        if group
    )


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


def _humanml_samples(dataset, maximum: int) -> list[tuple[int, dict[str, object]]]:
    selected: list[tuple[int, dict[str, object]]] = []
    for dataset_index in range(len(dataset)):
        sample = dataset[dataset_index]
        if str(sample["dataset"]) != "HumanML3D":
            continue
        selected.append((len(selected), sample))
        if maximum > 0 and len(selected) >= maximum:
            break
    if not selected:
        raise RuntimeError("the selected split contains no HumanML3D samples")
    return selected


def _result_contract(
    *,
    checkpoint: Path,
    weights: str,
    split: str,
    mode: str,
    sample_count: int,
    maximum_samples: int,
    maximum_frames: int,
    batch_size: int,
    base_seed: int,
    rolling_window_tokens: int,
    max_horizon_token: int,
    num_denoise_steps: int,
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_size": int(checkpoint.stat().st_size),
        "checkpoint_mtime_ns": int(checkpoint.stat().st_mtime_ns),
        "weights": weights,
        "split": split,
        "generation_mode": mode,
        "sample_count": int(sample_count),
        "maximum_samples": int(maximum_samples),
        "maximum_frames": int(maximum_frames),
        "batch_size": int(batch_size),
        "base_seed": int(base_seed),
        "rolling_window_tokens": int(rolling_window_tokens),
        "max_horizon_token": int(max_horizon_token),
        "num_denoise_steps": int(num_denoise_steps),
        "dense_xz": False,
        "precision": "bf16",
        "evaluator": "canonical_batched_stream_v1",
    }


def _read_completed_result(
    path: Path,
    *,
    contract: dict[str, Any],
    variant: GuidanceVariant,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("contract") != contract:
        raise RuntimeError(
            f"cannot resume {path}: its experiment contract does not match"
        )
    if result.get("variant") != asdict(variant):
        raise RuntimeError(f"cannot resume {path}: its CFG variant does not match")
    if not isinstance(result.get("metrics"), dict):
        raise RuntimeError(f"cannot resume {path}: metrics are missing")
    return result


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _evaluate_variant(
    module: LDFLightningModule,
    *,
    cfg,
    samples: list[tuple[int, dict[str, object]]],
    variant: GuidanceVariant,
    generation_mode: str,
    base_seed: int,
    maximum_frames: int,
    rolling_window_tokens: int,
    max_horizon_token: int,
    num_denoise_steps: int,
    progress_every: int,
    batch_size: int,
) -> tuple[dict[str, float], float]:
    del rolling_window_tokens, max_horizon_token
    batches = build_t2m_evaluation_batches(
        samples,
        maximum_frames=maximum_frames,
        batch_size=batch_size,
    )
    total = len(samples)
    next_progress = int(progress_every) if progress_every > 0 else total

    def progress(completed, local_total, current_batch_size, elapsed):
        nonlocal next_progress
        del local_total
        if completed == total or completed >= next_progress:
            eta = elapsed / completed * (total - completed)
            print(
                f"[{variant.name}] {completed}/{total} "
                f"batch={current_batch_size} "
                f"elapsed={_format_seconds(elapsed)} "
                f"eta={_format_seconds(eta)}",
                flush=True,
            )
            while next_progress <= completed:
                next_progress += max(1, int(progress_every))

    return evaluate_t2m_batches(
        module,
        metric_config=cfg.metrics.t2m,
        batches=batches,
        guidance_mode=variant.mode,
        cfg_scale_joint=variant.joint_scale,
        base_seed=base_seed,
        generation_mode=generation_mode,
        num_denoise_steps=num_denoise_steps,
        progress_callback=progress,
    )


def _evaluate_variant_group(
    *,
    config_path: str,
    overrides: list[str],
    checkpoint_path: str,
    output_root_path: str,
    device_index: int,
    variants: tuple[GuidanceVariant, ...],
    contract: dict[str, Any],
    split: str,
    maximum_samples: int,
    generation_mode: str,
    base_seed: int,
    maximum_frames: int,
    rolling_window_tokens: int,
    max_horizon_token: int,
    num_denoise_steps: int,
    progress_every: int,
    batch_size: int,
    raw_weights: bool,
) -> list[dict[str, Any]]:
    """Evaluate complete variants on one GPU without distributed collectives."""

    # Each process spends its heavy work on one GPU. Limiting CPU thread pools
    # avoids exhausting per-user threads when several independent workers start.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.cuda.set_device(int(device_index))
    device = torch.device("cuda", int(device_index))
    cfg = _load_config(config_path, overrides)
    dataset = create_dataset(cfg, split)
    samples = _humanml_samples(dataset, maximum_samples)
    if len(samples) != int(contract["sample_count"]):
        raise RuntimeError(
            f"GPU {device_index} selected {len(samples)} samples, expected "
            f"{contract['sample_count']}"
        )

    checkpoint = Path(checkpoint_path)
    output_root = Path(output_root_path)
    module = _load_module(cfg, checkpoint, device)
    weights = "raw" if raw_weights else "ema"
    results: list[dict[str, Any]] = []
    parameter_scope = nullcontext() if raw_weights else module.use_ema_parameters()
    with parameter_scope:
        for variant in variants:
            print(
                f"\n[gpu:{device_index}][{variant.name}] mode={variant.mode} "
                f"joint_scale={variant.joint_scale:g} "
                f"samples={len(samples)} weights={weights}",
                flush=True,
            )
            metrics, elapsed = _evaluate_variant(
                module,
                cfg=cfg,
                samples=samples,
                variant=variant,
                generation_mode=generation_mode,
                base_seed=base_seed,
                maximum_frames=maximum_frames,
                rolling_window_tokens=rolling_window_tokens,
                max_horizon_token=max_horizon_token,
                num_denoise_steps=num_denoise_steps,
                progress_every=progress_every,
                batch_size=batch_size,
            )
            result = {
                "contract": contract,
                "variant": asdict(variant),
                "metrics": metrics,
                "elapsed_seconds": float(elapsed),
                "worker_device": int(device_index),
            }
            write_json(output_root / f"{variant.name}.json", result)
            results.append(result)
            print(
                f"[gpu:{device_index}][{variant.name}] "
                f"FID={metrics['FID']:.4f} "
                f"elapsed={_format_seconds(elapsed)}",
                flush=True,
            )
    return results


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    metric_names = sorted(
        {
            str(name)
            for result in results
            for name in result.get("metrics", {})
        }
    )
    columns = [
        "name",
        "mode",
        "joint_scale",
        "elapsed_seconds",
        *metric_names,
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            variant = result["variant"]
            row = {
                "name": variant["name"],
                "mode": variant["mode"],
                "joint_scale": variant["joint_scale"],
                "elapsed_seconds": result["elapsed_seconds"],
                **result["metrics"],
            }
            writer.writerow(row)
    temporary.replace(path)


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\nT2M CFG sweep results", flush=True)
    print(
        f"{'variant':<16} {'scale':>7} {'FID':>10} "
        f"{'Diversity':>11} {'R@1':>9} {'time':>10}",
        flush=True,
    )
    for result in results:
        variant = result["variant"]
        metrics = result["metrics"]
        print(
            f"{variant['name']:<16} "
            f"{float(variant['joint_scale']):>7.2f} "
            f"{float(metrics.get('FID', float('nan'))):>10.4f} "
            f"{float(metrics.get('Diversity', float('nan'))):>11.4f} "
            f"{float(metrics.get('R_precision_top_1', float('nan'))):>9.4f} "
            f"{_format_seconds(float(result['elapsed_seconds'])):>10}",
            flush=True,
        )
    finite = [
        result
        for result in results
        if np.isfinite(float(result["metrics"].get("FID", float("nan"))))
    ]
    if finite:
        best = min(finite, key=lambda item: float(item["metrics"]["FID"]))
        print(
            f"best FID: {best['variant']['name']} = "
            f"{float(best['metrics']['FID']):.4f}",
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ldf.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--joint-scales",
        nargs="+",
        type=float,
        default=(1.0, 1.5, 2.0, 2.5, 3.0),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--generation-mode",
        choices=("stream", "rolling"),
        default="stream",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="0 evaluates the complete HumanML3D split; use a small value for smoke tests",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-horizon-token", type=int)
    parser.add_argument("--num-denoise-steps", type=int)
    parser.add_argument("--rolling-window-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Maximum equal-length T2M samples generated together on each GPU; "
            "defaults to data.val_batch_size"
        ),
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device",
        default=None,
        help="One torch device, for example cuda:0 or cpu",
    )
    device_group.add_argument(
        "--devices",
        nargs="+",
        type=int,
        help=(
            "Visible CUDA indices for variant-parallel evaluation, for example "
            "--devices 0 1 2 3"
        ),
    )
    parser.add_argument("--output", default="outputs/t2m_cfg_sweep")
    parser.add_argument(
        "--raw-weights",
        action="store_true",
        help="Use raw training weights instead of EMA weights",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed variants whose stored experiment contract matches",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    if args.generation_mode != "stream":
        raise ValueError(
            "batched T2M CFG sweep currently supports generation-mode=stream only"
        )

    cfg = _load_config(args.config, args.override)
    batch_size = (
        int(cfg.data.val_batch_size)
        if args.batch_size is None
        else int(args.batch_size)
    )
    if batch_size <= 0:
        raise ValueError("--batch-size/data.val_batch_size must be positive")
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if (device.type == "cuda" or args.devices) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.devices:
        available_devices = int(torch.cuda.device_count())
        invalid = [
            index
            for index in args.devices
            if index < 0 or index >= available_devices
        ]
        if invalid:
            raise ValueError(
                f"--devices contains unavailable visible CUDA indices {invalid}; "
                f"torch sees {available_devices} device(s)"
            )

    generation_cfg = cfg.validation.generation
    maximum_frames = (
        int(cfg.data.max_frames)
        if args.max_frames is None
        else int(args.max_frames)
    )
    maximum_frames -= maximum_frames % 4
    if maximum_frames <= 0:
        raise ValueError("--max-frames must include at least one four-frame token")
    max_horizon_token = (
        int(generation_cfg.max_horizon_token)
        if args.max_horizon_token is None
        else int(args.max_horizon_token)
    )
    num_denoise_steps = (
        int(generation_cfg.num_denoise_steps)
        if args.num_denoise_steps is None
        else int(args.num_denoise_steps)
    )
    rolling_window_tokens = (
        int(generation_cfg.rolling.window_tokens)
        if args.rolling_window_tokens is None
        else int(args.rolling_window_tokens)
    )
    base_seed = (
        int(cfg.validation.get("seed", cfg.get("seed", 0)))
        if args.seed is None
        else int(args.seed)
    )

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    variants = guidance_variants(args.joint_scales)
    dataset = create_dataset(cfg, args.split)
    samples = _humanml_samples(dataset, args.max_samples)
    sample_count = len(samples)
    weights = "raw" if args.raw_weights else "ema"
    contract = _result_contract(
        checkpoint=checkpoint,
        weights=weights,
        split=args.split,
        mode=args.generation_mode,
        sample_count=sample_count,
        maximum_samples=args.max_samples,
        maximum_frames=maximum_frames,
        batch_size=batch_size,
        base_seed=base_seed,
        rolling_window_tokens=rolling_window_tokens,
        max_horizon_token=max_horizon_token,
        num_denoise_steps=num_denoise_steps,
    )

    output_root = Path(args.output) / checkpoint.stem
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    pending: list[GuidanceVariant] = []
    for variant in variants:
        result_path = output_root / f"{variant.name}.json"
        completed = (
            _read_completed_result(
                result_path,
                contract=contract,
                variant=variant,
            )
            if args.resume
            else None
        )
        if completed is None:
            pending.append(variant)
        else:
            print(f"[{variant.name}] reusing {result_path}", flush=True)
            results.append(completed)

    if pending and args.devices:
        # Workers reopen the dataset themselves. Do not retain another complete
        # in-memory copy in the parent throughout the sweep.
        del samples
        del dataset
        groups = assign_variant_groups(pending, args.devices)
        assignments = ", ".join(
            f"cuda:{device_index}=[{', '.join(item.name for item in group)}]"
            for device_index, group in groups
        )
        print(f"parallel CFG assignments: {assignments}", flush=True)
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(groups),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _evaluate_variant_group,
                    config_path=str(args.config),
                    overrides=list(args.override),
                    checkpoint_path=str(checkpoint),
                    output_root_path=str(output_root),
                    device_index=device_index,
                    variants=group,
                    contract=contract,
                    split=args.split,
                    maximum_samples=args.max_samples,
                    generation_mode=args.generation_mode,
                    base_seed=base_seed,
                    maximum_frames=maximum_frames,
                    rolling_window_tokens=rolling_window_tokens,
                    max_horizon_token=max_horizon_token,
                    num_denoise_steps=num_denoise_steps,
                    progress_every=args.progress_every,
                    batch_size=batch_size,
                    raw_weights=bool(args.raw_weights),
                ): device_index
                for device_index, group in groups
            }
            for future in as_completed(futures):
                device_index = futures[future]
                try:
                    results.extend(future.result())
                except Exception as error:
                    raise RuntimeError(
                        f"CFG sweep worker on cuda:{device_index} failed"
                    ) from error
    elif pending:
        module = _load_module(cfg, checkpoint, device)
        parameter_scope = (
            nullcontext() if args.raw_weights else module.use_ema_parameters()
        )
        with parameter_scope:
            for variant in pending:
                print(
                    f"\n[{variant.name}] mode={variant.mode} "
                    f"joint_scale={variant.joint_scale:g} "
                    f"samples={len(samples)} weights={weights}",
                    flush=True,
                )
                metrics, elapsed = _evaluate_variant(
                    module,
                    cfg=cfg,
                    samples=samples,
                    variant=variant,
                    generation_mode=args.generation_mode,
                    base_seed=base_seed,
                    maximum_frames=maximum_frames,
                    rolling_window_tokens=rolling_window_tokens,
                    max_horizon_token=max_horizon_token,
                    num_denoise_steps=num_denoise_steps,
                    progress_every=args.progress_every,
                    batch_size=batch_size,
                )
                result = {
                    "contract": contract,
                    "variant": asdict(variant),
                    "metrics": metrics,
                    "elapsed_seconds": float(elapsed),
                }
                write_json(output_root / f"{variant.name}.json", result)
                results.append(result)
                print(
                    f"[{variant.name}] FID={metrics['FID']:.4f} "
                    f"elapsed={_format_seconds(elapsed)}",
                    flush=True,
                )

    result_order = {
        variant.name: index for index, variant in enumerate(variants)
    }
    results.sort(key=lambda item: result_order[item["variant"]["name"]])
    finite_results = [
        result
        for result in results
        if np.isfinite(float(result["metrics"].get("FID", float("nan"))))
    ]
    best = (
        None
        if not finite_results
        else min(
            finite_results,
            key=lambda item: float(item["metrics"]["FID"]),
        )["variant"]["name"]
    )
    aggregate = {
        "contract": contract,
        "best_fid_variant": best,
        "results": results,
    }
    write_json(output_root / "summary.json", aggregate)
    _write_csv(output_root / "summary.csv", results)
    _print_summary(results)
    print(f"wrote T2M CFG sweep to {output_root}", flush=True)


if __name__ == "__main__":
    main()
