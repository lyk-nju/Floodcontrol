"""Standalone Dataset runner for BodyVAE reconstruction evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from datasets.babel import BABELDataset
from datasets.humanml3d import HumanML3DDataset
from models.vae_wan_1d import BodyVAE
from utils.training.vae.checkpoint import load_vae_checkpoint

from .artifacts import save_sample_outputs, validate_model_id, write_json
from .metrics import mean_metrics, reconstruction_metrics
from .reconstruction import (
    ROLLING_PROTOCOL,
    STREAM_PROTOCOL,
    load_motion_sample,
    rolling_reconstruct,
    stream_reconstruct,
)


DATASET_TYPES = {
    "humanml3d": HumanML3DDataset,
    "babel": BABELDataset,
}


def evaluate_dataset(
    model: BodyVAE,
    *,
    dataset_name: str,
    dataset_config=None,
    dataset=None,
    sample_count: int,
    output_root: Path,
    model_id: str,
    device: torch.device,
    parity_atol: float,
    render_video: bool,
    render_fps: int,
    mode: str,
    window_config=None,
) -> dict[str, object]:
    """Evaluate one Dataset source and write its reconstruction manifest."""

    if dataset is None:
        if dataset_name not in DATASET_TYPES:
            raise ValueError(f"unsupported VAE evaluation dataset {dataset_name!r}")
        if dataset_config is None:
            raise ValueError("dataset_config is required when dataset is not provided")
        dataset = DATASET_TYPES[dataset_name](
            meta_paths=[dataset_config.val_meta_path],
            split="val",
            artifact_path=dataset_config.artifact_path,
            text_path=dataset_config.get("text_path"),
            fps=model.fps,
        )
    if len(dataset) < sample_count:
        raise RuntimeError(
            f"{dataset_name} val split contains only {len(dataset)} samples, "
            f"expected {sample_count}"
        )
    manifest_samples = []
    all_metrics = []
    for index in range(sample_count):
        sample = load_motion_sample(dataset[index], expected_fps=model.fps)
        if mode == "stream":
            result = stream_reconstruct(
                model, sample, device=device, parity_atol=parity_atol
            )
        elif mode == "rolling":
            if window_config is None:
                raise ValueError("rolling evaluation requires window configuration")
            result = rolling_reconstruct(
                model,
                sample,
                device=device,
                history_tokens=int(window_config.history_tokens),
                commit_tokens=int(window_config.commit_tokens),
                parity_atol=parity_atol,
            )
        else:
            raise ValueError(f"unsupported VAE reconstruction mode {mode!r}")
        metrics = reconstruction_metrics(sample, result)
        outputs = save_sample_outputs(
            sample,
            result,
            metrics,
            output_root=output_root,
            dataset_name=dataset_name,
            model_id=model_id,
            render_video=render_video,
            render_fps=render_fps,
        )
        all_metrics.append(metrics)
        manifest_samples.append(
            {
                "index": index,
                "sample_id": sample.sample_id,
                "source_dataset": sample.dataset,
                "frames": int(sample.body_motion.shape[0]),
                "outputs": outputs,
            }
        )
        print(
            f"[{dataset_name} {index + 1}/{sample_count}] {sample.sample_id}: "
            f"position={metrics['position_mae_m']:.6f}m, "
            f"rotation={metrics['rotation_geodesic_deg']:.4f}deg"
        )
    dataset_output = output_root / dataset_name / validate_model_id(model_id)
    summary = {
        "dataset": dataset_name,
        "model_id": model_id,
        "protocol": all_metrics[0]["protocol"],
        "sample_count": sample_count,
        "mean_metrics": mean_metrics(all_metrics),
    }
    write_json(dataset_output / "manifest.json", {"samples": manifest_samples})
    write_json(dataset_output / "summary.json", summary)
    return summary


def _load_model(cfg, device: torch.device) -> tuple[BodyVAE, dict[str, object]]:
    model = load_vae_checkpoint(
        cfg.model.checkpoint_path,
        model_params=OmegaConf.to_container(cfg.model.params, resolve=True),
    )
    return model.to(device), {"path": str(cfg.model.checkpoint_path), "weights": "ema"}


def run(cfg, *, mode: str) -> dict[str, object]:
    """Run configured stream or rolling evaluation across all Dataset sources."""

    device = torch.device(str(cfg.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA evaluation requested but unavailable: {device}")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    sample_count = int(cfg.sample_count)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    output_root = Path(str(cfg.output_dir))
    model_id = validate_model_id(str(cfg.model.model_id))
    model, checkpoint_metadata = _load_model(cfg, device)
    checkpoint_metadata["model_id"] = model_id
    dataset_summaries = {}
    for dataset_name, dataset_config in cfg.datasets.items():
        dataset_summaries[dataset_name] = evaluate_dataset(
            model,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            sample_count=sample_count,
            output_root=output_root,
            model_id=model_id,
            device=device,
            parity_atol=float(cfg.stream_parity_atol),
            render_video=bool(cfg.render.enabled),
            render_fps=int(cfg.render.fps),
            mode=mode,
            window_config=cfg.get("window"),
        )
    protocol = STREAM_PROTOCOL if mode == "stream" else ROLLING_PROTOCOL
    summary = {
        "model_id": model_id,
        "protocol": protocol,
        "mode": mode,
        "root_policy": "source explicit root shared by original and reconstruction",
        "latent_policy": (
            "deterministic raw posterior mu with no latent whitening"
        ),
        "checkpoint": checkpoint_metadata,
        "datasets": dataset_summaries,
    }
    if mode == "rolling":
        summary["window"] = OmegaConf.to_container(cfg.window, resolve=True)
    write_json(output_root / "summaries" / f"{model_id}.json", summary)
    return summary


def load_task_config(default_config: str):
    """Load one VAE evaluation YAML and apply CLI-only runtime overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--device")
    parser.add_argument("--output")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.device:
        cfg.device = args.device
    if args.output:
        cfg.output_dir = args.output
    if args.sample_count is not None:
        cfg.sample_count = args.sample_count
    if args.skip_video:
        cfg.render.enabled = False
    return cfg


__all__ = ["evaluate_dataset", "load_task_config", "run"]
