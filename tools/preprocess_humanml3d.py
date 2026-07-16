"""Build root5/body265 artifacts from the processed HumanML3D dataset.

The builder reads ``new_joint_vecs`` and source split TXT files from one
HumanML3D root, then writes a separate, resumable ``HumanML3D_motion`` dataset.
Rotations are explicitly identified as HumanML3D IK-derived rotations rather
than native AMASS/SMPL pose parameters.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from tools.convert_motion_263_to_265 import HUMANML_DIM
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    aligned_frame_floor,
    require_aligned_frame_count,
)
from tools.build_motion_artifact import (
    artifact_is_current,
    atomic_copy,
    atomic_write_text as _atomic_write_text,
    process_file,
)


DEFAULT_SPLITS = ("train", "val", "test")


def _convert_task(task: tuple[str, str, float]) -> tuple[str, int]:
    source_value, target_value, fps = task
    source, target = Path(source_value), Path(target_value)
    if artifact_is_current(source, target, fps=fps):
        return "skipped", 0
    result = process_file(source, target, fps=fps)
    return "converted", int(result["frames"])


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"HumanML3D split metadata file not found at {path}")
    names: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        name = line.strip()
        if not name:
            continue
        if Path(name).name != name:
            raise ValueError(
                f"sample id must be a plain filename stem: {path}:{line_number}"
            )
        names.append(name)
    if not names:
        raise RuntimeError(f"HumanML3D split contains no sample ids: {path}")
    return names


def _inspect_motion(path: Path) -> tuple[int, bool]:
    feature = np.load(path, mmap_mode="r", allow_pickle=False)
    if feature.ndim != 2 or feature.shape[-1] != HUMANML_DIM:
        raise ValueError(
            f"HumanML3D source must be [F,{HUMANML_DIM}], got {feature.shape} at {path}"
        )
    usable = aligned_frame_floor(int(feature.shape[0]))
    return usable, bool(np.isfinite(feature).all())


def build_dataset(
    source_root: Path,
    output: Path,
    *,
    splits: Iterable[str] = DEFAULT_SPLITS,
    artifact_path: str = "artifacts",
    workers: int = 1,
    fps: float = 20.0,
    min_frames: int = 20,
    skip_missing: bool = False,
) -> dict[str, object]:
    source_root, output = Path(source_root), Path(output)
    motion_root = source_root / "new_joint_vecs"
    text_root = source_root / "texts"
    if not motion_root.is_dir():
        raise RuntimeError(
            f"HUMANML3D_DATA_REQUIRED: 263D source directory not found at {motion_root}"
        )
    if not text_root.is_dir():
        raise RuntimeError(f"HumanML3D text directory not found at {text_root}")
    if output.resolve() == source_root.resolve():
        raise ValueError(
            "output must not be the legacy HumanML3D root because source split "
            "TXT files would be overwritten"
        )
    artifact_subdir = Path(artifact_path)
    if artifact_subdir.is_absolute() or ".." in artifact_subdir.parts:
        raise ValueError("artifact_path must be a relative directory")
    artifact_root = output / artifact_subdir

    min_frames = int(min_frames)
    if min_frames < FRAMES_PER_TOKEN:
        raise ValueError("min_frames must be a positive multiple of four")
    require_aligned_frame_count(min_frames)

    source_split_names: dict[str, list[str]] = {}
    missing_by_split: dict[str, list[str]] = {}
    for split in splits:
        split = str(split)
        if split not in DEFAULT_SPLITS:
            raise ValueError(f"unsupported split {split!r}")
        names = _read_split(source_root / f"{split}.txt")
        missing = [name for name in names if not (motion_root / f"{name}.npy").is_file()]
        if missing and not skip_missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"HUMANML3D_DATA_REQUIRED: {len(missing)} {split} motions are missing; "
                f"first entries: {preview}"
            )
        missing_set = set(missing)
        source_split_names[split] = [name for name in names if name not in missing_set]
        missing_by_split[split] = missing

    motion_inspection: dict[str, tuple[int, bool]] = {}
    for names in source_split_names.values():
        for name in names:
            if name not in motion_inspection:
                motion_inspection[name] = _inspect_motion(motion_root / f"{name}.npy")
    invalid_by_split = {
        split: [name for name in names if not motion_inspection[name][1]]
        for split, names in source_split_names.items()
    }
    too_short_by_split = {
        split: [
            name for name in names
            if motion_inspection[name][1] and motion_inspection[name][0] < min_frames
        ]
        for split, names in source_split_names.items()
    }
    split_names = {
        split: [
            name for name in names
            if motion_inspection[name][1] and motion_inspection[name][0] >= min_frames
        ]
        for split, names in source_split_names.items()
    }
    unique_names: dict[str, None] = {}
    for names in split_names.values():
        for name in names:
            unique_names[name] = None

    tasks = [
        (
            str(motion_root / f"{name}.npy"),
            str(artifact_root / f"{name}.npz"),
            float(fps),
        )
        for name in unique_names
    ]
    converted = skipped = total_frames = 0
    workers = max(1, int(workers))
    if workers == 1:
        results = map(_convert_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_convert_task, tasks, chunksize=8)
    try:
        for index, (status, frames) in enumerate(results, start=1):
            converted += status == "converted"
            skipped += status == "skipped"
            total_frames += frames
            if index % 500 == 0 or index == len(tasks):
                print(
                    f"processed {index}/{len(tasks)} artifacts "
                    f"(converted={converted}, skipped={skipped})",
                    flush=True,
                )
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=True)

    output.mkdir(parents=True, exist_ok=True)
    for name in unique_names:
        source_text = text_root / f"{name}.txt"
        if not source_text.is_file():
            raise RuntimeError(f"HumanML3D text file not found at {source_text}")
        atomic_copy(source_text, output / "texts" / source_text.name)
    for split, names in split_names.items():
        _atomic_write_text(
            output / f"{split}.txt", "".join(f"{name}\n" for name in names)
        )
    _atomic_write_text(
        output / "all.txt", "".join(f"{name}\n" for name in unique_names)
    )
    summary = {
        "source_root": str(source_root.resolve()),
        "artifact_path": str(artifact_subdir),
        "fps": float(fps),
        "min_frames": min_frames,
        "splits": {name: len(values) for name, values in split_names.items()},
        "all": len(unique_names),
        "missing": {name: len(values) for name, values in missing_by_split.items()},
        "invalid_nonfinite": {
            name: len(values) for name, values in invalid_by_split.items()
        },
        "too_short": {name: len(values) for name, values in too_short_by_split.items()},
        "unique_artifacts": len(tasks),
        "copied_texts": len(unique_names),
        "converted": converted,
        "skipped": skipped,
        "converted_frames": total_frames,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        required=True,
        help="legacy HumanML3D root containing new_joint_vecs and split TXT files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="separate processed dataset root, for example HumanML3D_motion",
    )
    parser.add_argument(
        "--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=DEFAULT_SPLITS
    )
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="write filtered target splits instead of failing on missing source motions",
    )
    args = parser.parse_args()
    summary = build_dataset(
        Path(args.source_root),
        Path(args.output),
        splits=args.splits,
        artifact_path=args.artifact_path,
        workers=args.workers,
        fps=args.fps,
        min_frames=args.min_frames,
        skip_missing=args.skip_missing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
