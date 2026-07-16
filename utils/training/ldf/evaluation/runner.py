"""Scheduled training-time generation evaluation for the hybrid LDF."""

from __future__ import annotations

import hashlib
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info

from metrics.humanml import convert_root5_body265_to_humanml263
from metrics.stream import compute_stream_boundary_metrics
from metrics.t2m import T2MMetrics
from metrics.trajectory import (
    compute_dense_xz_metrics,
    compute_foot_skating_ratio,
    summarize_dense_xz_records,
)
from utils.training.ldf.data import create_dataset

from .artifacts import (
    evaluation_artifact_dirs,
    render_comparison_video,
    save_dense_xz_sample,
    write_json,
)
from .generation import GENERATION_MODES, generate_evaluation_sequence


def _stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(base_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "little") % (2**63 - 1)


def _scalar_metrics(summary: dict[str, Any], prefix: str) -> dict[str, float]:
    output = {}
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if np.isfinite(number):
                output[f"{prefix}/{key}"] = number
    return output


def _frame_count(sample: dict[str, object], maximum: int) -> int:
    frames = min(int(sample["root_motion"].shape[0]), int(maximum))
    frames -= frames % 4
    if frames <= 0:
        raise ValueError("evaluation sample has no complete four-frame token")
    return frames


class LDFEvaluationRunner:
    """Own heavy generation evaluation without expanding the Lightning module."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._dataset = None
        self._probe_datasets: dict[str, object] = {}

    @property
    def enabled(self) -> bool:
        validation = self.cfg.get("validation") or {}
        generation = validation.get("generation") or {}
        t2m = validation.get("t2m") or {}
        dense = validation.get("dense_xz") or {}
        return bool(generation.get("enabled", False)) and (
            bool(t2m.get("enabled", False)) or bool(dense.get("enabled", False))
        )

    def _validation_dataset(self):
        if self._dataset is None:
            self._dataset = create_dataset(self.cfg, "val")
        return self._dataset

    def _probe_dataset(self, probe: str):
        probe = str(probe)
        if probe not in self._probe_datasets:
            probe_paths = self.cfg.data.get("test_probe_meta_paths") or {}
            if probe not in probe_paths:
                raise RuntimeError(
                    f"data.test_probe_meta_paths does not define probe {probe!r}"
                )
            self._probe_datasets[probe] = create_dataset(
                self.cfg,
                "test",
                meta_paths=list(probe_paths[probe]),
            )
        return self._probe_datasets[probe]

    def _selected_samples(
        self,
        *,
        dataset=None,
        limit: int,
        dataset_name: str | None = None,
        sample_ids: list[str] | None = None,
    ) -> list[dict[str, object]]:
        requested = set(str(value) for value in (sample_ids or []))
        selected = []
        source = self._validation_dataset() if dataset is None else dataset
        for index in range(len(source)):
            sample = source[index]
            if dataset_name is not None and str(sample["dataset"]) != dataset_name:
                continue
            if requested and str(sample["name"]) not in requested:
                continue
            selected.append(sample)
            if limit > 0 and len(selected) >= limit:
                break
        if requested:
            found = {str(sample["name"]) for sample in selected}
            missing = requested - found
            if missing:
                raise RuntimeError(
                    f"evaluation sample ids were not found: {sorted(missing)}"
                )
        if not selected:
            raise RuntimeError("generation evaluation selected no validation samples")
        return selected

    @staticmethod
    def _generation_config(validation) -> dict[str, Any]:
        generation = validation.generation
        modes = tuple(str(mode) for mode in generation.modes)
        if not modes or any(mode not in GENERATION_MODES for mode in modes):
            raise ValueError("validation.generation.modes must contain stream/rolling")
        return {
            "modes": modes,
            "num_runs": int(generation.get("num_runs", 1)),
            "num_denoise_steps": int(generation.num_denoise_steps),
            "max_horizon_token": int(generation.max_horizon_token),
            "rolling_window_tokens": int(generation.rolling.window_tokens),
            "rebase_on_roll": bool(generation.rolling.get("rebase_on_roll", True)),
            "render": bool(generation.get("render", True)),
        }

    def _log(self, module, metrics: dict[str, float], *, step: int) -> None:
        if metrics and module.logger is not None:
            module.logger.log_metrics(metrics, step=int(step))

    def _log_videos(
        self,
        module,
        *,
        paths: list[Path],
        key: str,
        step: int,
    ) -> None:
        if not paths or module.logger is None:
            return
        try:
            import wandb

            module.logger.experiment.log(
                {
                    key: [wandb.Video(str(path), format="mp4") for path in paths],
                    "trainer/global_step": int(step),
                },
                step=int(step),
            )
        except Exception as error:
            warnings.warn(f"could not log validation videos: {error}", stacklevel=2)

    def _run_dense_xz(self, module, *, step: int, step_tag: str) -> None:
        validation = self.cfg.validation
        config = self._generation_config(validation)
        dense = validation.dense_xz
        probe = str(dense.probe)
        samples = self._selected_samples(
            dataset=self._probe_dataset(probe),
            limit=0,
        )
        segment_frames = int(dense.get("segment_frames", 20))
        video_samples = int(dense.get("video_samples", len(samples)))
        maximum_frames = int(self.cfg.data.max_frames)
        base_seed = int(validation.get("seed", self.cfg.get("seed", 0)))

        for mode in config["modes"]:
            by_dataset: dict[str, list[dict[str, Any]]] = {}
            videos: list[Path] = []
            for sample_index, sample in enumerate(samples):
                frames = _frame_count(sample, maximum_frames)
                target_root = sample["root_motion"][:frames]
                target_body = sample["body_motion"][:frames]
                for run_index in range(config["num_runs"]):
                    seed = _stable_seed(
                        base_seed,
                        sample["dataset"],
                        sample["name"],
                        run_index,
                    )
                    generated = generate_evaluation_sequence(
                        module,
                        sample,
                        mode=mode,
                        seed=seed,
                        frame_count=frames,
                        dense_xz=True,
                        rolling_window_tokens=config["rolling_window_tokens"],
                        max_horizon_token=config["max_horizon_token"],
                        num_denoise_steps=config["num_denoise_steps"],
                        rebase_on_roll=config["rebase_on_roll"],
                    )
                    record = compute_dense_xz_metrics(
                        generated.root_motion,
                        target_root,
                        segment_frames=segment_frames,
                    )
                    record["foot_skating_ratio"] = compute_foot_skating_ratio(
                        generated.root_motion,
                        generated.body_motion,
                        fps=float(module.model.fps),
                    )
                    boundaries = compute_stream_boundary_metrics(
                        generated.root_motion,
                        generated.body_motion,
                        list(range(4, frames + 1, 4)),
                    )
                    record.update(
                        {
                            "root_boundary_jump_mean": boundaries["root_jump_mean"],
                            "joint_boundary_jump_mean": boundaries["joint_jump_mean"],
                            "dataset": str(sample["dataset"]),
                            "name": str(sample["name"]),
                            "mode": mode,
                            "run_index": int(run_index),
                            "seed": int(seed),
                            "caption": generated.prompt.caption,
                        }
                    )
                    by_dataset.setdefault(str(sample["dataset"]), []).append(record)
                    sample_id = str(sample["name"])
                    if config["num_runs"] > 1:
                        sample_id = f"{sample_id}_run{run_index}"
                    probe = f"dense_xz_{mode}"
                    should_render = bool(config["render"] and sample_index < video_samples and run_index == 0)
                    dirs = save_dense_xz_sample(
                        save_dir=self.cfg.save_dir,
                        dataset=str(sample["dataset"]),
                        probe=probe,
                        step_tag=step_tag,
                        sample_id=sample_id,
                        caption=generated.prompt.caption,
                        normalized_root=generated.normalized_motion.root_motion[0],
                        normalized_latent=generated.normalized_motion.latent_motion[0],
                        predicted_root=generated.root_motion,
                        predicted_body=generated.body_motion,
                        target_root=target_root,
                        target_body=target_body,
                        trajectory_mask=torch.ones(frames, dtype=torch.bool),
                        prompt_change_frames=generated.prompt.change_frames,
                        record=record,
                        render=False,
                        fps=float(module.model.fps),
                    )
                    if should_render:
                        try:
                            render_comparison_video(
                                target_root=target_root,
                                target_body=target_body,
                                predicted_root=generated.root_motion,
                                predicted_body=generated.body_motion,
                                predicted_video_path=dirs["video"] / f"{sample_id}.mp4",
                                composite_path=dirs["composite"] / f"{sample_id}.mp4",
                                caption=generated.prompt.caption,
                                fps=float(module.model.fps),
                            )
                            videos.append(dirs["composite"] / f"{sample_id}.mp4")
                        except Exception as error:
                            warnings.warn(
                                f"dense XZ artifacts saved but video rendering failed for "
                                f"{sample_id}: {error}",
                                stacklevel=2,
                            )

            for dataset, records in by_dataset.items():
                summary = summarize_dense_xz_records(records)
                summary.update({"mode": mode, "probe": "dense_xz"})
                metric_dir = evaluation_artifact_dirs(
                    self.cfg.save_dir,
                    dataset,
                    f"dense_xz_{mode}",
                    step_tag,
                )["metrics"]
                write_json(
                    metric_dir / "summary.json",
                    {"summary": summary, "samples": records},
                )
                self._log(
                    module,
                    _scalar_metrics(
                        summary, f"eval/dense_xz/{mode}/{dataset}"
                    ),
                    step=step,
                )
                rank_zero_info(
                    f"[dense-xz][{mode}][{dataset}][{step_tag}] "
                    f"ADE={summary.get('ade_mean', float('nan')):.4f} "
                    f"FDE={summary.get('fde_mean', float('nan')):.4f}"
                )
            self._log_videos(
                module,
                paths=videos,
                key=f"eval/dense_xz/{mode}/videos",
                step=step,
            )

    def _run_t2m(self, module, *, step: int, step_tag: str) -> None:
        validation = self.cfg.validation
        config = self._generation_config(validation)
        samples = self._selected_samples(
            limit=0,
            dataset_name="HumanML3D",
        )
        maximum_frames = int(self.cfg.data.max_frames)
        base_seed = int(validation.get("seed", self.cfg.get("seed", 0)))

        for mode in config["modes"]:
            metric = T2MMetrics(self.cfg.metrics.t2m)
            metric.eval()
            for sample in samples:
                frames = _frame_count(sample, maximum_frames)
                seed = _stable_seed(
                    base_seed,
                    "t2m",
                    sample["dataset"],
                    sample["name"],
                )
                generated = generate_evaluation_sequence(
                    module,
                    sample,
                    mode=mode,
                    seed=seed,
                    frame_count=frames,
                    dense_xz=False,
                    rolling_window_tokens=config["rolling_window_tokens"],
                    max_horizon_token=config["max_horizon_token"],
                    num_denoise_steps=config["num_denoise_steps"],
                    rebase_on_roll=config["rebase_on_roll"],
                )
                reference = convert_root5_body265_to_humanml263(
                    sample["root_motion"][:frames],
                    sample["body_motion"][:frames],
                    tail="drop",
                ).detach().cpu()
                predicted = convert_root5_body265_to_humanml263(
                    generated.root_motion,
                    generated.body_motion,
                    tail="drop",
                ).detach().cpu()
                length = int(reference.shape[0])
                metric.update(
                    reference[None],
                    predicted[None],
                    [length],
                    [length],
                    [list(generated.prompt.tokens)],
                )
            values = metric.compute(False)
            summary = {
                key: float(value.detach().cpu().item())
                for key, value in values.items()
            }
            summary.update(
                {
                    "num_samples": len(samples),
                    "mode": mode,
                    "split": "val",
                }
            )
            metric_dir = evaluation_artifact_dirs(
                self.cfg.save_dir,
                "HumanML3D",
                f"t2m_{mode}",
                step_tag,
            )["metrics"]
            write_json(metric_dir / "summary.json", {"summary": summary})
            self._log(
                module,
                _scalar_metrics(summary, f"eval/t2m/{mode}/HumanML3D"),
                step=step,
            )
            rank_zero_info(
                f"[t2m][{mode}][{step_tag}] "
                f"FID={summary.get('FID', float('nan')):.4f}"
            )

    def maybe_run(self, module) -> None:
        if not self.enabled:
            return
        step = int(module.global_step)
        if step <= 0:
            return
        validation = self.cfg.validation
        generation_due = bool(validation.dense_xz.get("enabled", False)) and (
            step % int(validation.generation.steps) == 0
        )
        t2m_due = bool(validation.t2m.get("enabled", False)) and (
            step % int(validation.t2m.steps) == 0
        )
        if not generation_due and not t2m_due:
            return
        step_tag = f"step_{step:06d}"

        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        model_training = bool(module.model.training)
        vae_training = bool(module.vae.training)
        try:
            evaluation_seed = int(validation.get("seed", self.cfg.get("seed", 0)))
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed % (2**32))
            torch.manual_seed(evaluation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(evaluation_seed)
            module.model.eval()
            module.vae.eval()
            with module.use_ema_parameters():
                if generation_due:
                    self._run_dense_xz(module, step=step, step_tag=step_tag)
                if t2m_due:
                    self._run_t2m(module, step=step, step_tag=step_tag)
        finally:
            module.model.train(model_training)
            module.vae.train(vae_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


__all__ = ["LDFEvaluationRunner"]
