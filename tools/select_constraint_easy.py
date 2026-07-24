#!/usr/bin/env python3
"""Select low-information, near-stationary XZ trajectories for LDF training.

The output split is intentionally conservative: a sample is constraint-easy
only when its complete processed root trajectory stays inside a small spatial
region, ends near its starting point, and does not accumulate a long path.
These samples can then use a higher XZ constraint-drop probability while still
participating in ordinary text-to-motion supervision.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_FPS = 20.0
DEFAULT_MAX_PATH_LENGTH_M = 1.5
DEFAULT_MAX_DISPLACEMENT_M = 0.35
DEFAULT_MAX_SPATIAL_EXTENT_M = 0.50
DEFAULT_MOVING_SPEED_MPS = 0.10


@dataclass(frozen=True)
class TrajectoryMetrics:
    frames: int
    duration_seconds: float
    path_length_m: float
    displacement_m: float
    spatial_extent_m: float
    max_radius_from_start_m: float
    mean_speed_mps: float
    p95_speed_mps: float
    moving_ratio: float


@dataclass(frozen=True)
class ConstraintEasyThresholds:
    max_path_length_m: float = DEFAULT_MAX_PATH_LENGTH_M
    max_displacement_m: float = DEFAULT_MAX_DISPLACEMENT_M
    max_spatial_extent_m: float = DEFAULT_MAX_SPATIAL_EXTENT_M

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


def load_root_motion(path: str | Path) -> np.ndarray:
    """Load and validate physical root5 from one processed motion artifact."""

    artifact = Path(path)
    with np.load(artifact, allow_pickle=False) as data:
        if "root_motion" not in data:
            raise ValueError(f"root_motion is missing from {artifact}")
        root = np.asarray(data["root_motion"])
    if root.dtype != np.float32:
        raise ValueError(f"root_motion must be float32 in {artifact}")
    if root.ndim != 2 or root.shape[-1] != 5 or root.shape[0] <= 0:
        raise ValueError(f"root_motion must be non-empty [F,5] in {artifact}")
    if not np.isfinite(root).all():
        raise ValueError(f"root_motion contains non-finite values in {artifact}")
    return root


def compute_trajectory_metrics(
    root_motion: np.ndarray,
    *,
    fps: float = DEFAULT_FPS,
    moving_speed_mps: float = DEFAULT_MOVING_SPEED_MPS,
) -> TrajectoryMetrics:
    """Compute translation/rotation-invariant XZ motion statistics."""

    root = np.asarray(root_motion)
    if root.ndim != 2 or root.shape[-1] != 5 or root.shape[0] <= 0:
        raise ValueError("root_motion must be non-empty [F,5]")
    if not np.isfinite(root).all():
        raise ValueError("root_motion contains non-finite values")
    fps = float(fps)
    moving_speed_mps = float(moving_speed_mps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(moving_speed_mps) or moving_speed_mps < 0.0:
        raise ValueError("moving_speed_mps must be finite and non-negative")

    xz = root[:, [0, 2]].astype(np.float64, copy=False)
    frames = int(len(xz))
    if frames == 1:
        return TrajectoryMetrics(
            frames=1,
            duration_seconds=0.0,
            path_length_m=0.0,
            displacement_m=0.0,
            spatial_extent_m=0.0,
            max_radius_from_start_m=0.0,
            mean_speed_mps=0.0,
            p95_speed_mps=0.0,
            moving_ratio=0.0,
        )

    steps = np.linalg.norm(np.diff(xz, axis=0), axis=-1)
    speeds = steps * fps
    # Maximum pairwise distance is a true spatial diameter and therefore
    # remains unchanged by global translation or yaw rotation. HumanML clips
    # are short enough that the dense pairwise calculation is inexpensive.
    pairwise = xz[:, None, :] - xz[None, :, :]
    extent = np.linalg.norm(pairwise, axis=-1).max()
    radius = np.linalg.norm(xz - xz[0], axis=-1)
    return TrajectoryMetrics(
        frames=frames,
        duration_seconds=float((frames - 1) / fps),
        path_length_m=float(steps.sum()),
        displacement_m=float(np.linalg.norm(xz[-1] - xz[0])),
        spatial_extent_m=float(extent),
        max_radius_from_start_m=float(radius.max()),
        mean_speed_mps=float(speeds.mean()),
        p95_speed_mps=float(np.quantile(speeds, 0.95)),
        moving_ratio=float(np.mean(speeds >= moving_speed_mps)),
    )


def constraint_easy_score(
    metrics: TrajectoryMetrics,
    thresholds: ConstraintEasyThresholds,
) -> float:
    """Return the maximum normalized threshold ratio (easy iff <= 1)."""

    ratios = []
    for value, limit in (
        (metrics.path_length_m, thresholds.max_path_length_m),
        (metrics.displacement_m, thresholds.max_displacement_m),
        (metrics.spatial_extent_m, thresholds.max_spatial_extent_m),
    ):
        if limit == 0.0:
            ratios.append(0.0 if value == 0.0 else math.inf)
        else:
            ratios.append(float(value) / float(limit))
    return float(max(ratios))


def is_constraint_easy(
    metrics: TrajectoryMetrics,
    thresholds: ConstraintEasyThresholds,
) -> bool:
    """Whether a complete sample has low-information, near-stationary XZ."""

    return constraint_easy_score(metrics, thresholds) <= 1.0


def _read_sample_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"metadata split not found: {path}")
    names: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        name = line.strip()
        if not name:
            continue
        if Path(name).name != name:
            raise ValueError(f"sample id must be a filename stem: {path}:{line_number}")
        if name in seen:
            raise ValueError(f"duplicate sample id {name!r} in {path}")
        seen.add(name)
        names.append(name)
    if not names:
        raise RuntimeError(f"metadata split contains no sample ids: {path}")
    return names


def _measure_artifact(arguments: tuple[str, str, float, float]):
    name, artifact_path, fps, moving_speed_mps = arguments
    metrics = compute_trajectory_metrics(
        load_root_motion(artifact_path),
        fps=fps,
        moving_speed_mps=moving_speed_mps,
    )
    return name, metrics


def _measure_all(
    names: list[str],
    artifact_dir: Path,
    *,
    fps: float,
    moving_speed_mps: float,
    workers: int,
) -> list[tuple[str, TrajectoryMetrics]]:
    arguments = [
        (name, str(artifact_dir / f"{name}.npz"), fps, moving_speed_mps)
        for name in names
    ]
    missing = [
        artifact_path
        for _, artifact_path, _, _ in arguments
        if not Path(artifact_path).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} motion artifacts are missing; first entries: {preview}"
        )
    if workers <= 1:
        iterator: Iterable[tuple[str, TrajectoryMetrics]] = map(
            _measure_artifact, arguments
        )
        return list(iterator)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_measure_artifact, arguments, chunksize=32))


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(np.quantile(array, quantile))
        for name, quantile in (
            ("min", 0.0),
            ("p10", 0.1),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("max", 1.0),
        )
    }


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True, help="Processed training split TXT")
    parser.add_argument("--output", required=True, help="Output constraint-easy TXT")
    parser.add_argument(
        "--summary",
        help="Optional JSON report; defaults beside --output",
    )
    parser.add_argument(
        "--artifact-path",
        default="artifacts",
        help="Artifact directory relative to the split file",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--max-path-length-m",
        type=float,
        default=DEFAULT_MAX_PATH_LENGTH_M,
    )
    parser.add_argument(
        "--max-displacement-m",
        type=float,
        default=DEFAULT_MAX_DISPLACEMENT_M,
    )
    parser.add_argument(
        "--max-spatial-extent-m",
        type=float,
        default=DEFAULT_MAX_SPATIAL_EXTENT_M,
    )
    parser.add_argument(
        "--moving-speed-mps",
        type=float,
        default=DEFAULT_MOVING_SPEED_MPS,
        help="Diagnostic threshold only; it does not decide membership",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    meta_path = Path(args.meta).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    summary_path = (
        output_path.with_suffix(".summary.json")
        if args.summary is None
        else Path(args.summary).expanduser().resolve()
    )
    artifact_path = Path(args.artifact_path)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise ValueError("artifact-path must be relative to the metadata directory")
    workers = int(args.workers)
    if workers <= 0:
        raise ValueError("workers must be positive")
    thresholds = ConstraintEasyThresholds(
        max_path_length_m=float(args.max_path_length_m),
        max_displacement_m=float(args.max_displacement_m),
        max_spatial_extent_m=float(args.max_spatial_extent_m),
    )

    names = _read_sample_ids(meta_path)
    measured = _measure_all(
        names,
        meta_path.parent / artifact_path,
        fps=float(args.fps),
        moving_speed_mps=float(args.moving_speed_mps),
        workers=workers,
    )
    records = []
    selected_names = []
    for name, metrics in measured:
        score = constraint_easy_score(metrics, thresholds)
        selected = score <= 1.0
        if selected:
            selected_names.append(name)
        records.append(
            {
                "name": name,
                "constraint_easy": selected,
                "score": score,
                **asdict(metrics),
            }
        )

    metric_names = tuple(asdict(measured[0][1]))
    summary = {
        "contract": "constraint_easy_v1",
        "meta": str(meta_path),
        "artifact_dir": str((meta_path.parent / artifact_path).resolve()),
        "output": str(output_path),
        "fps": float(args.fps),
        "moving_speed_mps_diagnostic": float(args.moving_speed_mps),
        "thresholds": asdict(thresholds),
        "counts": {
            "total": len(names),
            "constraint_easy": len(selected_names),
            "trajectory_teaching": len(names) - len(selected_names),
        },
        "constraint_easy_ratio": len(selected_names) / len(names),
        "quantiles": {
            metric_name: _quantiles(
                [float(getattr(metrics, metric_name)) for _, metrics in measured]
            )
            for metric_name in metric_names
        },
        "records": records,
    }
    _atomic_write_text(
        output_path,
        "".join(f"{name}\n" for name in selected_names),
    )
    _atomic_write_text(
        summary_path,
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    print(
        f"selected {len(selected_names)}/{len(names)} "
        f"({len(selected_names) / len(names):.1%}) constraint-easy samples"
    )
    print(f"wrote split: {output_path}")
    print(f"wrote report: {summary_path}")


if __name__ == "__main__":
    main()
