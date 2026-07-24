"""Scheduled training-time generation evaluation for the hybrid LDF."""

from __future__ import annotations

import hashlib
import random
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from lightning.pytorch.utilities import rank_zero_info

from metrics.stream import compute_stream_boundary_metrics
from metrics.trajectory import (
    compute_dense_xz_metrics,
    compute_foot_skating_ratio,
    summarize_dense_xz_records,
)
from utils.training.ldf.data import create_dataset
from utils.training.ldf.metrics import (
    compute_heading_metrics,
    compute_rollout_heading_metrics,
)

from .artifacts import (
    evaluation_artifact_dirs,
    render_comparison_video,
    save_dense_xz_sample,
    write_json,
)
from .generation import (
    GENERATION_MODES,
    compile_evaluation_prompt,
    generate_evaluation_sequence,
    rotate_evaluation_sample,
)
from .t2m import (
    T2MEvaluationBatch,
    evaluate_t2m_batches,
)


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


_ROLLOUT_LOG_NAMES = {
    "cold_root_deg": "val/cold/root_deg",
    "cold_root_anti": "val/cold/root_anti",
    "cold_body_deg": "val/cold/body_deg",
    "cold_feet_deg": "val/cold/feet_deg",
    "roll_root_deg": "val/roll/root_deg",
    "roll_root_p95": "val/roll/root_p95",
    "roll_root_anti": "val/roll/root_anti",
    "roll_body_deg": "val/roll/body_deg",
    "roll_feet_deg": "val/roll/feet_deg",
    "roll_body_rel": "val/roll/body_rel",
    "roll_feet_rel": "val/roll/feet_rel",
    "roll_feet_rev": "val/roll/feet_rev",
    "ade": "val/roll/ade",
    "fde": "val/roll/fde",
}


def _finite_values(records: list[dict[str, Any]], key: str) -> np.ndarray:
    values = np.asarray(
        [float(record[key]) for record in records if key in record],
        dtype=np.float64,
    )
    return values[np.isfinite(values)]


def _compact_rollout_metrics(
    records: list[dict[str, Any]],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for record_key, log_name in _ROLLOUT_LOG_NAMES.items():
        values = _finite_values(records, record_key)
        if values.size:
            output[log_name] = float(values.mean())
    return output


def _standard_case_metrics(
    records: list[dict[str, Any]],
    *,
    case_name: str,
) -> dict[str, float]:
    case_records = [
        record for record in records if str(record.get("name")) == str(case_name)
    ]
    if not case_records:
        raise RuntimeError(f"standard validation case {case_name!r} was not generated")
    definitions = {
        "cold_mean": ("cold_root_deg", "mean"),
        "cold_max": ("cold_root_max", "max"),
        "root_mean": ("roll_root_deg", "mean"),
        "root_max": ("roll_root_max", "max"),
        "body_rel_mean": ("roll_body_rel", "mean"),
        "body_rel_max": ("roll_body_rel_max", "max"),
        "feet_rel_mean": ("roll_feet_rel", "mean"),
        "feet_rel_max": ("roll_feet_rel_max", "max"),
    }
    output: dict[str, float] = {}
    for log_suffix, (record_key, reduction) in definitions.items():
        values = _finite_values(case_records, record_key)
        if not values.size:
            continue
        value = values.mean() if reduction == "mean" else values.max()
        output[f"val/case/{case_name}/{log_suffix}"] = float(value)
    return output


def _format_t2m_summary(
    summary: dict[str, Any],
    *,
    mode: str,
    step_tag: str,
) -> str:
    """Format the complete rank-zero T2M result for the training console."""

    def metric(name: str) -> str | None:
        value = summary.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        number = float(value)
        return f"{number:.4f}" if np.isfinite(number) else None

    lines = [
        f"[t2m][{mode}][{step_tag}] "
        f"samples={int(summary.get('num_samples', 0))} "
        f"cfg={summary.get('cfg_mode', 'unknown')} "
        f"scale={float(summary.get('cfg_scale_joint', 1.0)):.2f}"
    ]

    fid = metric("FID")
    if fid is not None:
        lines.append(f"  FID={fid}")

    matching = metric("Matching_score")
    gt_matching = metric("gt_Matching_score")
    if matching is not None or gt_matching is not None:
        values = []
        if matching is not None:
            values.append(f"generated={matching}")
        if gt_matching is not None:
            values.append(f"ground_truth={gt_matching}")
        lines.append(f"  MatchingScore: {', '.join(values)}")

    r_precision = []
    gt_r_precision = []
    for top_k in range(1, 4):
        value = metric(f"R_precision_top_{top_k}")
        if value is not None:
            r_precision.append(f"top{top_k}={value}")
        gt_value = metric(f"gt_R_precision_top_{top_k}")
        if gt_value is not None:
            gt_r_precision.append(f"top{top_k}={gt_value}")
    if r_precision:
        lines.append(f"  R-Precision: {', '.join(r_precision)}")
    if gt_r_precision:
        lines.append(f"  GT R-Precision: {', '.join(gt_r_precision)}")

    diversity = metric("Diversity")
    gt_diversity = metric("gt_Diversity")
    if diversity is not None or gt_diversity is not None:
        values = []
        if diversity is not None:
            values.append(f"generated={diversity}")
        if gt_diversity is not None:
            values.append(f"ground_truth={gt_diversity}")
        lines.append(f"  Diversity: {', '.join(values)}")
    return "\n".join(lines)


def _frame_count(sample: dict[str, object], maximum: int) -> int:
    frames = min(int(sample["root_motion"].shape[0]), int(maximum))
    frames -= frames % 4
    if frames <= 0:
        raise ValueError("evaluation sample has no complete four-frame token")
    return frames


def _distributed_rank_world(module) -> tuple[int, int]:
    """Return the active process identity without requiring DDP in tests."""

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())
    return int(getattr(module, "global_rank", 0)), max(
        1, int(getattr(module, "world_size", 1))
    )


def _all_gather_objects(value: object) -> list[object]:
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    gathered: list[object] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast_object(value: object, *, source: int = 0) -> object:
    if not dist.is_available() or not dist.is_initialized():
        return value
    payload = [value if dist.get_rank() == int(source) else None]
    dist.broadcast_object_list(payload, src=int(source))
    return payload[0]


def _distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _peek_sample_identity(source, index: int) -> tuple[str, str] | None:
    """Read source/name metadata without loading a complete motion artifact."""

    if isinstance(source, (list, tuple)):
        item = source[index]
        if isinstance(item, dict) and "dataset" in item and "name" in item:
            return str(item["dataset"]), str(item["name"])
        return None
    samples = getattr(source, "samples", None)
    if isinstance(samples, list) and index < len(samples):
        entry = samples[index]
        if isinstance(entry, tuple) and len(entry) == 2:
            nested_source, nested_index = entry
            return _peek_sample_identity(nested_source, int(nested_index))
    records = getattr(source, "dataset", None)
    if isinstance(records, list) and index < len(records):
        record = records[index]
        if isinstance(record, dict) and "dataset" in record and "name" in record:
            return str(record["dataset"]), str(record["name"])
    return None


class LDFEvaluationRunner:
    """Own heavy generation evaluation without expanding the Lightning module."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._dataset = None
        self._probe_datasets: dict[str, object] = {}
        self._t2m_batch_plans: dict[
            tuple[int, int, str],
            tuple[tuple[int, tuple[tuple[int, int], ...]], ...],
        ] = {}
        self._text_coverage_validated = False
        self._startup_evaluation_completed = False

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
    ) -> list[dict[str, object]]:
        selected = []
        source = self._validation_dataset() if dataset is None else dataset
        for index in range(len(source)):
            sample = source[index]
            if dataset_name is not None and str(sample["dataset"]) != dataset_name:
                continue
            selected.append(sample)
            if limit > 0 and len(selected) >= limit:
                break
        if not selected:
            raise RuntimeError("generation evaluation selected no validation samples")
        return selected

    def _selected_sample_shard(
        self,
        module,
        *,
        dataset=None,
        limit: int,
        dataset_name: str | None = None,
        sample_names: tuple[str, ...] | None = None,
    ) -> tuple[list[tuple[int, dict[str, object]]], int]:
        """Select globally, but load only the samples owned by this DDP rank."""

        rank, world_size = _distributed_rank_world(module)
        selected: list[tuple[int, dict[str, object]]] = []
        selected_count = 0
        source = self._validation_dataset() if dataset is None else dataset
        for source_index in range(len(source)):
            identity = _peek_sample_identity(source, source_index)
            sample = None
            if identity is None:
                sample = source[source_index]
                identity = (str(sample["dataset"]), str(sample["name"]))
            source_name, sample_name = identity
            if dataset_name is not None and source_name != dataset_name:
                continue
            if sample_names is not None and sample_name not in sample_names:
                continue
            global_index = selected_count
            if global_index % world_size == rank:
                if sample is None:
                    sample = source[source_index]
                selected.append((global_index, sample))
            selected_count += 1
            if limit > 0 and selected_count >= limit:
                break
        if selected_count == 0:
            raise RuntimeError("generation evaluation selected no validation samples")
        return selected, selected_count

    def _t2m_batch_shard(
        self,
        module,
        *,
        maximum_frames: int,
        batch_size: int,
        dataset_name: str,
    ) -> tuple[tuple[T2MEvaluationBatch, ...], int]:
        """Build global fixed batches once, then assign complete batches to ranks."""

        if int(batch_size) <= 0:
            raise ValueError("validation T2M batch_size must be positive")
        rank, world_size = _distributed_rank_world(module)
        key = (int(maximum_frames), int(batch_size), str(dataset_name))
        plan = self._t2m_batch_plans.get(key)
        source = self._validation_dataset()

        if plan is None:
            rank_zero_plan = None
            if rank == 0:
                try:
                    buckets: dict[int, list[tuple[int, int]]] = {}
                    selected_count = 0
                    for source_index in range(len(source)):
                        identity = _peek_sample_identity(source, source_index)
                        sample = None
                        if identity is None:
                            sample = source[source_index]
                            identity = (
                                str(sample["dataset"]),
                                str(sample["name"]),
                            )
                        if identity[0] != str(dataset_name):
                            continue
                        if sample is None:
                            sample = source[source_index]
                        frames = _frame_count(sample, maximum_frames)
                        buckets.setdefault(frames, []).append(
                            (selected_count, source_index)
                        )
                        selected_count += 1
                    if selected_count == 0:
                        raise RuntimeError(
                            "generation evaluation selected no T2M validation samples"
                        )
                    batches = []
                    for frames in sorted(buckets, reverse=True):
                        values = buckets[frames]
                        for start in range(0, len(values), int(batch_size)):
                            batches.append(
                                (
                                    int(frames),
                                    tuple(
                                        values[start : start + int(batch_size)]
                                    ),
                                )
                            )
                    rank_zero_plan = tuple(batches)
                except Exception as error:
                    rank_zero_plan = {
                        "error": (
                            "rank zero could not construct the T2M batch plan: "
                            f"{type(error).__name__}: {error}"
                        )
                    }
            plan = _broadcast_object(rank_zero_plan)
            if isinstance(plan, dict) and "error" in plan:
                raise RuntimeError(str(plan["error"]))
            if not isinstance(plan, tuple):
                raise RuntimeError("distributed T2M batch plan was not broadcast")
            self._t2m_batch_plans[key] = plan

        local_batches = []
        for batch_index, (frames, references) in enumerate(plan):
            if batch_index % world_size != rank:
                continue
            local_batches.append(
                T2MEvaluationBatch(
                    frame_count=int(frames),
                    samples=tuple(
                        (int(global_index), source[int(source_index)])
                        for global_index, source_index in references
                    ),
                )
            )
        sample_count = sum(len(references) for _, references in plan)
        return tuple(local_batches), sample_count

    def validate_text_coverage(self, module) -> None:
        """Fail before training when scheduled evaluation prompts are not encoded."""

        if self._text_coverage_validated or not self.enabled:
            return
        validation = self.cfg.validation
        required = {""}
        if bool(validation.dense_xz.get("enabled", False)):
            probe = str(validation.dense_xz.probe)
            maximum_frames = int(self.cfg.data.max_frames)
            for sample in self._selected_samples(
                dataset=self._probe_dataset(probe),
                limit=0,
            ):
                frames = _frame_count(sample, maximum_frames)
                required.update(
                    compile_evaluation_prompt(sample, frame_count=frames).timeline
                )
        try:
            module.text_embeddings.lookup(sorted(required))
        except KeyError as error:
            raise RuntimeError(
                "EVALUATION_TEXT_EMBEDDINGS_INCOMPLETE: scheduled validation "
                "prompts are missing from data.text_embeddings_path. Regenerate "
                "the table with tools/pretokenize_t5_text.py --reuse-existing."
            ) from error
        self._text_coverage_validated = True

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
            "render": bool(generation.get("render", True)),
        }

    @staticmethod
    def _yaw_video_config(dense) -> tuple[float, ...]:
        yaw_degrees = tuple(
            float(value) for value in dense.get("video_yaw_degrees", (0, 90, 180))
        )
        if not yaw_degrees or any(not np.isfinite(value) for value in yaw_degrees):
            raise ValueError(
                "validation.dense_xz.video_yaw_degrees must contain finite values"
            )
        if len(set(yaw_degrees)) != len(yaw_degrees):
            raise ValueError("validation.dense_xz.video_yaw_degrees must be unique")
        return yaw_degrees

    def _render_yaw_videos(
        self,
        module,
        *,
        sample: dict[str, object],
        base_generated,
        base_target_root: torch.Tensor,
        base_target_body: torch.Tensor,
        frame_count: int,
        seed: int,
        mode: str,
        config: dict[str, Any],
        yaw_degrees: tuple[float, ...],
        video_dir: Path,
        composite_dir: Path,
    ) -> list[str]:
        """Render fixed-yaw variants with paired root/latent source noise."""

        paths = []
        sample_id = str(sample["name"])
        for angle in yaw_degrees:
            if abs(angle) < 1e-12:
                generated = base_generated
                target_root = base_target_root
                target_body = base_target_body
            else:
                rotated_sample = rotate_evaluation_sample(
                    sample,
                    frame_count=frame_count,
                    yaw_degrees=angle,
                )
                target_root = rotated_sample["root_motion"]
                target_body = rotated_sample["body_motion"]
                generated = generate_evaluation_sequence(
                    module,
                    rotated_sample,
                    mode=mode,
                    seed=seed,
                    frame_count=frame_count,
                    dense_xz=True,
                    rolling_window_tokens=config["rolling_window_tokens"],
                    max_horizon_token=config["max_horizon_token"],
                    num_denoise_steps=config["num_denoise_steps"],
                    initial_noise_yaw_degrees=angle,
                )
            rounded = round(angle)
            if abs(angle - rounded) < 1e-8:
                yaw_label = f"yaw_{rounded:03d}deg"
            else:
                yaw_label = f"yaw_{angle:07.3f}deg".replace(".", "p")
            output_name = f"{sample_id}_{yaw_label}.mp4"
            predicted_path = video_dir / output_name
            composite_path = composite_dir / output_name
            render_comparison_video(
                target_root=target_root,
                target_body=target_body,
                predicted_root=generated.root_motion,
                predicted_body=generated.body_motion,
                predicted_video_path=predicted_path,
                composite_path=composite_path,
                caption=f"{generated.prompt.caption} | yaw {angle:g} deg",
                fps=float(module.model.fps),
            )
            paths.append(str(composite_path))
        return paths

    def _log(self, module, metrics: dict[str, float], *, step: int) -> None:
        rank, _ = _distributed_rank_world(module)
        if rank == 0 and metrics and module.logger is not None:
            module.logger.log_metrics(metrics, step=int(step))

    def _log_videos(
        self,
        module,
        *,
        paths: list[Path],
        key: str,
        step: int,
    ) -> None:
        rank, _ = _distributed_rank_world(module)
        if rank != 0 or not paths or module.logger is None:
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

    @contextmanager
    def _evaluation_context(self, module):
        """Run generation with deterministic RNG and EMA, then restore training."""

        validation = self.cfg.validation
        evaluation_seed = int(validation.get("seed", self.cfg.get("seed", 0)))
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_device = module.device if module.device.type == "cuda" else None
        cuda_state = (
            torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None
        )
        model_training = bool(module.model.training)
        vae_training = bool(module.vae.training)
        try:
            random.seed(evaluation_seed)
            np.random.seed(evaluation_seed % (2**32))
            torch.random.default_generator.manual_seed(evaluation_seed)
            if cuda_device is not None:
                torch.cuda.manual_seed(evaluation_seed)
            module.model.eval()
            module.vae.eval()
            with module.use_ema_parameters():
                yield
        finally:
            module.model.train(model_training)
            module.vae.train(vae_training)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, cuda_device)

    def _run_startup_yaw_videos(self, module, *, step: int, step_tag: str) -> None:
        """Render only the fixed paired-yaw samples at fit startup."""

        validation = self.cfg.validation
        config = self._generation_config(validation)
        dense = validation.dense_xz
        yaw_degrees = self._yaw_video_config(dense)
        standard_cases = tuple(
            str(name)
            for name in dense.get("standard_cases", ("000021", "001168"))
        )
        probe = str(dense.probe)
        samples, _ = self._selected_sample_shard(
            module,
            dataset=self._probe_dataset(probe),
            limit=0,
            sample_names=standard_cases,
        )
        maximum_frames = int(self.cfg.data.max_frames)
        base_seed = int(validation.get("seed", self.cfg.get("seed", 0)))

        for mode in config["modes"]:
            local_videos: list[str] = []
            for _, sample in samples:
                frames = _frame_count(sample, maximum_frames)
                seed = _stable_seed(
                    base_seed,
                    sample["dataset"],
                    sample["name"],
                    0,
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
                    initial_noise_yaw_degrees=0.0,
                )
                dirs = evaluation_artifact_dirs(
                    self.cfg.save_dir,
                    str(sample["dataset"]),
                    f"dense_xz_{mode}",
                    step_tag,
                )
                dirs["video"].mkdir(parents=True, exist_ok=True)
                dirs["composite"].mkdir(parents=True, exist_ok=True)
                try:
                    local_videos.extend(
                        self._render_yaw_videos(
                            module,
                            sample=sample,
                            base_generated=generated,
                            base_target_root=sample["root_motion"][:frames],
                            base_target_body=sample["body_motion"][:frames],
                            frame_count=frames,
                            seed=seed,
                            mode=mode,
                            config=config,
                            yaw_degrees=yaw_degrees,
                            video_dir=dirs["video"],
                            composite_dir=dirs["composite"],
                        )
                    )
                except Exception as error:
                    warnings.warn(
                        "startup paired-yaw video rendering failed for "
                        f"{sample['name']}: {error}",
                        stacklevel=2,
                    )
            gathered_videos = _all_gather_objects(local_videos)
            videos = [
                Path(path)
                for rank_paths in gathered_videos
                for path in rank_paths
            ]
            videos.sort(key=str)
            self._log_videos(
                module,
                paths=videos,
                key=f"eval/dense_xz/{mode}/startup_videos",
                step=step,
            )
            _distributed_barrier()

    def run_at_start(self, module) -> bool:
        """Consume the optional pre-fit validation with generation and T2M."""

        if self._startup_evaluation_completed or not self.enabled:
            return False
        validation = self.cfg.validation
        generation = validation.generation
        startup_enabled = (
            bool(self.cfg.get("train", False))
            and bool(generation.get("run_at_start", False))
        )
        yaw_due = (
            startup_enabled
            and bool(generation.get("render", True))
            and bool(validation.dense_xz.get("enabled", False))
        )
        t2m_due = startup_enabled and bool(validation.t2m.get("enabled", False))
        if not yaw_due and not t2m_due:
            self._startup_evaluation_completed = True
            return False
        step = int(module.global_step)
        _distributed_barrier()
        with self._evaluation_context(module):
            if yaw_due:
                self._run_startup_yaw_videos(
                    module,
                    step=step,
                    step_tag="fit_start",
                )
            if t2m_due:
                self._run_t2m(
                    module,
                    step=step,
                    step_tag="fit_start",
                )
        self._startup_evaluation_completed = True
        _distributed_barrier()
        return True

    def _run_dense_xz(self, module, *, step: int, step_tag: str) -> None:
        validation = self.cfg.validation
        config = self._generation_config(validation)
        dense = validation.dense_xz
        probe = str(dense.probe)
        standard_cases = tuple(
            str(name)
            for name in dense.get("standard_cases", ("000021", "001168"))
        )
        if len(set(standard_cases)) != len(standard_cases):
            raise ValueError("validation.dense_xz.standard_cases must be unique")
        samples, _ = self._selected_sample_shard(
            module,
            dataset=self._probe_dataset(probe),
            limit=0,
        )
        video_yaw_degrees = self._yaw_video_config(dense)
        maximum_frames = int(self.cfg.data.max_frames)
        base_seed = int(validation.get("seed", self.cfg.get("seed", 0)))
        rank, _ = _distributed_rank_world(module)

        for mode in config["modes"]:
            local_records: list[dict[str, Any]] = []
            local_videos: list[str] = []
            local_case_videos: list[str] = []
            for _, sample in samples:
                frames = _frame_count(sample, maximum_frames)
                target_root = sample["root_motion"][:frames]
                target_body = sample["body_motion"][:frames]
                for run_index in range(config["num_runs"]):
                    # Dense-XZ uses the same three fixed sources for every
                    # sample so checkpoints are compared under paired noise.
                    # T2M retains its independent stable-hash seeding below.
                    seed = base_seed + int(run_index)
                    generated = generate_evaluation_sequence(
                        module,
                        sample,
                        mode=mode,
                        guidance_mode="joint",
                        seed=seed,
                        frame_count=frames,
                        dense_xz=True,
                        rolling_window_tokens=config["rolling_window_tokens"],
                        max_horizon_token=config["max_horizon_token"],
                        num_denoise_steps=config["num_denoise_steps"],
                        initial_noise_yaw_degrees=(
                            0.0 if config["render"] and run_index == 0 else None
                        ),
                    )
                    record = compute_dense_xz_metrics(
                        generated.root_motion,
                        target_root,
                    )
                    record["foot_skating_ratio"] = compute_foot_skating_ratio(
                        generated.root_motion,
                        generated.body_motion,
                        fps=float(module.model.fps),
                    )
                    heading = compute_heading_metrics(
                        predicted_root=generated.root_motion[None],
                        target_root=target_root.to(generated.root_motion)[None],
                        predicted_body=generated.body_motion[None],
                        target_body=target_body.to(generated.body_motion)[None],
                        frame_mask=torch.ones(
                            1,
                            frames,
                            device=generated.root_motion.device,
                            dtype=torch.bool,
                        ),
                        fps=float(module.model.fps),
                    )
                    record.update(
                        {
                            name: float(value.detach().cpu())
                            for name, value in heading.items()
                        }
                    )
                    ability = compute_rollout_heading_metrics(
                        predicted_root=generated.root_motion[None],
                        target_root=target_root.to(generated.root_motion)[None],
                        predicted_body=generated.body_motion[None],
                        target_body=target_body.to(generated.body_motion)[None],
                        frame_mask=torch.ones(
                            1,
                            frames,
                            device=generated.root_motion.device,
                            dtype=torch.bool,
                        ),
                    )
                    record.update(
                        {
                            name: float(value.detach().cpu())
                            for name, value in ability.items()
                        }
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
                    local_records.append(record)
                    sample_id = str(sample["name"])
                    if config["num_runs"] > 1:
                        sample_id = f"{sample_id}_run{run_index}"
                    probe = f"dense_xz_{mode}"
                    dirs = save_dense_xz_sample(
                        save_dir=self.cfg.save_dir,
                        dataset=str(sample["dataset"]),
                        probe=probe,
                        step_tag=step_tag,
                        sample_id=sample_id,
                        caption=generated.prompt.caption,
                        root_motion=generated.hybrid_motion.root_motion[0],
                        latent_motion=generated.hybrid_motion.latent_motion[0],
                        predicted_root=generated.root_motion,
                        predicted_body=generated.body_motion,
                        target_root=target_root,
                        target_body=target_body,
                        trajectory_mask=torch.ones(frames, dtype=torch.bool),
                        prompt_change_frames=generated.prompt.change_frames,
                        record=record,
                        render=(
                            config["render"]
                            and mode == "stream"
                            and str(sample["name"]) in standard_cases
                        ),
                        fps=float(module.model.fps),
                    )
                    if (
                        config["render"]
                        and mode == "stream"
                        and str(sample["name"]) in standard_cases
                    ):
                        local_case_videos.append(
                            str(dirs["composite"] / f"{sample_id}.mp4")
                        )
                    if (
                        config["render"]
                        and run_index == 0
                        and str(sample["name"]) in standard_cases
                    ):
                        try:
                            local_videos.extend(
                                self._render_yaw_videos(
                                    module,
                                    sample=sample,
                                    base_generated=generated,
                                    base_target_root=target_root,
                                    base_target_body=target_body,
                                    frame_count=frames,
                                    seed=seed,
                                    mode=mode,
                                    config=config,
                                    yaw_degrees=video_yaw_degrees,
                                    video_dir=dirs["video"],
                                    composite_dir=dirs["composite"],
                                )
                            )
                        except Exception as error:
                            warnings.warn(
                                "dense XZ artifacts saved but paired yaw video rendering "
                                f"failed for {sample_id}: {error}",
                                stacklevel=2,
                            )

            gathered_records = _all_gather_objects(local_records)
            gathered_videos = _all_gather_objects(local_videos)
            gathered_case_videos = _all_gather_objects(local_case_videos)
            if rank == 0:
                all_records = [
                    record
                    for rank_records in gathered_records
                    for record in rank_records
                ]
                by_dataset: dict[str, list[dict[str, Any]]] = {}
                for record in all_records:
                    by_dataset.setdefault(str(record["dataset"]), []).append(record)
                for dataset, records in by_dataset.items():
                    records.sort(
                        key=lambda value: (
                            str(value["name"]), int(value["run_index"])
                        )
                    )
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
                    rank_zero_info(
                        f"[dense-xz][{mode}][{dataset}][{step_tag}] "
                        f"ADE={summary.get('ade_mean', float('nan')):.4f} "
                        f"FDE={summary.get('fde_mean', float('nan')):.4f} "
                        f"root-GT-root="
                        f"{summary.get('root_gt_root_heading_angle_deg_mean', float('nan')):.2f}deg "
                        f"body-GT-body="
                        f"{summary.get('body_gt_body_heading_angle_deg_mean', float('nan')):.2f}deg "
                        f"feet-GT-feet="
                        f"{summary.get('feet_gt_feet_heading_angle_deg_mean', float('nan')):.2f}deg "
                        f"root-body-heading="
                        f"{summary.get('root_body_heading_angle_deg_mean', float('nan')):.2f}deg "
                        f"root-feet-heading="
                        f"{summary.get('root_feet_heading_angle_deg_mean', float('nan')):.2f}deg"
                    )
                if mode == "stream":
                    compact = _compact_rollout_metrics(all_records)
                    for case_name in standard_cases:
                        compact.update(
                            _standard_case_metrics(
                                all_records,
                                case_name=case_name,
                            )
                        )
                    self._log(module, compact, step=step)
                    case_summary_path = (
                        Path(self.cfg.save_dir)
                        / "validation"
                        / "cases"
                        / step_tag
                        / "summary.json"
                    )
                    write_json(
                        case_summary_path,
                        {
                            "mode": mode,
                            "seeds": [
                                base_seed + run_index
                                for run_index in range(config["num_runs"])
                            ],
                            "metrics": compact,
                            "cases": list(standard_cases),
                        },
                    )
                videos = [
                    Path(path)
                    for rank_paths in gathered_videos
                    for path in rank_paths
                ]
                videos.sort(key=str)
                self._log_videos(
                    module,
                    paths=videos,
                    key=f"val/roll/{mode}_videos",
                    step=step,
                )
                case_videos = [
                    Path(path)
                    for rank_paths in gathered_case_videos
                    for path in rank_paths
                ]
                case_videos.sort(key=str)
                if mode == "stream":
                    self._log_videos(
                        module,
                        paths=case_videos,
                        key="val/case/videos",
                        step=step,
                    )
            _distributed_barrier()

    def _run_t2m(self, module, *, step: int, step_tag: str) -> None:
        validation = self.cfg.validation
        config = self._generation_config(validation)
        guidance_mode = str(validation.t2m.get("cfg_mode", "nocfg"))
        cfg_scale_joint = float(
            validation.t2m.get(
                "cfg_scale_joint",
                module.model.cfg_scale_joint,
            )
        )
        batch_size = int(self.cfg.data.val_batch_size)
        batches, sample_count = self._t2m_batch_shard(
            module,
            maximum_frames=int(self.cfg.data.max_frames),
            batch_size=batch_size,
            dataset_name="HumanML3D",
        )
        base_seed = int(validation.get("seed", self.cfg.get("seed", 0)))
        rank, _ = _distributed_rank_world(module)

        for mode in config["modes"]:
            summary, _ = evaluate_t2m_batches(
                module,
                metric_config=self.cfg.metrics.t2m,
                batches=batches,
                guidance_mode=guidance_mode,
                cfg_scale_joint=cfg_scale_joint,
                base_seed=base_seed,
                generation_mode=mode,
                num_denoise_steps=config["num_denoise_steps"],
            )
            summary.update(
                {
                    "num_samples": sample_count,
                    "mode": mode,
                    "cfg_mode": guidance_mode,
                    "cfg_scale_joint": cfg_scale_joint,
                    "batch_size": batch_size,
                    "precision": "bf16",
                    "split": "val",
                }
            )
            if rank == 0:
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
                    _format_t2m_summary(
                        summary,
                        mode=mode,
                        step_tag=step_tag,
                    )
                )
            _distributed_barrier()

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
        _distributed_barrier()

        with self._evaluation_context(module):
            if generation_due:
                self._run_dense_xz(module, step=step, step_tag=step_tag)
            if t2m_due:
                self._run_t2m(module, step=step, step_tag=step_tag)


__all__ = ["LDFEvaluationRunner"]
