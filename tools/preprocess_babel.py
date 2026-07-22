"""Build root5/body259 artifacts from BABEL_streamed motions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.build_motion_artifact import (
    artifact_is_current,
    atomic_copy,
    atomic_write_text,
    process_file,
)
from tools.convert_motion_263_to_259 import HUMANML_DIM
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    aligned_frame_floor,
    require_aligned_frame_count,
)


DEFAULT_SPLIT_FILES = {
    "train": "train_processed.txt",
    "val": "val_processed.txt",
}


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"BABEL split metadata file not found at {path}")
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
        raise RuntimeError(f"BABEL split contains no sample ids: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"BABEL split contains duplicate sample ids: {path}")
    return names


def _inspect_motion(path: Path) -> tuple[int, bool]:
    feature = np.load(path, mmap_mode="r", allow_pickle=False)
    if feature.ndim != 2 or feature.shape[-1] != HUMANML_DIM:
        raise ValueError(
            f"BABEL motion must be [F,{HUMANML_DIM}], got {feature.shape} at {path}"
        )
    usable = aligned_frame_floor(int(feature.shape[0]))
    return usable, bool(np.isfinite(feature).all())


def _convert_task(task: tuple[str, str, float]) -> tuple[str, int]:
    source_value, target_value, fps = task
    source, target = Path(source_value), Path(target_value)
    if artifact_is_current(source, target, fps=fps):
        return "skipped", 0
    result = process_file(source, target, fps=fps)
    return "converted", int(result["frames"])


def build_dataset(
    source_root: Path,
    output: Path,
    *,
    split_files: Mapping[str, str] | None = None,
    motion_path: str = "motions",
    artifact_path: str = "artifacts",
    workers: int = 1,
    fps: float = 20.0,
    min_frames: int = 20,
    skip_missing: bool = False,
) -> dict[str, object]:
    source_root, output = Path(source_root), Path(output)
    motion_subdir = Path(motion_path)
    artifact_subdir = Path(artifact_path)
    for name, value in (("motion_path", motion_subdir), ("artifact_path", artifact_subdir)):
        if value.is_absolute() or ".." in value.parts:
            raise ValueError(f"{name} must be a relative directory")
    motion_root = source_root / motion_subdir
    text_root = source_root / "texts"
    if not motion_root.is_dir():
        raise RuntimeError(f"BABEL_DATA_REQUIRED: motion directory not found at {motion_root}")
    if not text_root.is_dir():
        raise RuntimeError(f"BABEL text directory not found at {text_root}")
    if output.resolve() == source_root.resolve():
        raise ValueError("output must be separate from the legacy BABEL_streamed root")

    min_frames = int(min_frames)
    if min_frames < FRAMES_PER_TOKEN:
        raise ValueError("min_frames must be a positive multiple of four")
    require_aligned_frame_count(min_frames)
    split_files = dict(split_files or DEFAULT_SPLIT_FILES)
    if not split_files:
        raise ValueError("at least one BABEL split must be configured")
    unsupported = set(split_files) - {"train", "val", "test"}
    if unsupported:
        raise ValueError(f"unsupported target splits: {sorted(unsupported)}")

    source_split_names: dict[str, list[str]] = {}
    missing_by_split: dict[str, list[str]] = {}
    for split, source_name in split_files.items():
        source_path = Path(source_name)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise ValueError("BABEL split paths must be relative to source_root")
        names = _read_split(source_root / source_path)
        missing = [name for name in names if not (motion_root / f"{name}.npy").is_file()]
        if missing and not skip_missing:
            preview = ", ".join(missing[:5])
            raise RuntimeError(
                f"BABEL_DATA_REQUIRED: {len(missing)} {split} motions are missing; "
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

    artifact_root = output / artifact_subdir
    tasks = [
        (
            str(motion_root / f"{name}.npy"),
            str(artifact_root / f"{name}.npz"),
            float(fps),
        )
        for name in unique_names
    ]
    converted = skipped = converted_frames = 0
    workers = max(1, int(workers))
    executor = None
    if workers == 1:
        results = map(_convert_task, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        results = executor.map(_convert_task, tasks, chunksize=8)
    try:
        for index, (status, frames) in enumerate(results, start=1):
            converted += status == "converted"
            skipped += status == "skipped"
            converted_frames += frames
            if index % 500 == 0 or index == len(tasks):
                print(
                    f"processed {index}/{len(tasks)} BABEL artifacts "
                    f"(converted={converted}, skipped={skipped})",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    output.mkdir(parents=True, exist_ok=True)
    for name in unique_names:
        source_text = text_root / f"{name}.txt"
        if not source_text.is_file():
            raise RuntimeError(f"BABEL text file not found at {source_text}")
        atomic_copy(source_text, output / "texts" / source_text.name)
    for split, names in split_names.items():
        atomic_write_text(output / f"{split}.txt", "".join(f"{name}\n" for name in names))
    atomic_write_text(
        output / "all.txt", "".join(f"{name}\n" for name in unique_names)
    )
    summary = {
        "source_dataset": "BABEL_streamed",
        "source_root": str(source_root.resolve()),
        "motion_path": str(motion_subdir),
        "artifact_path": str(artifact_subdir),
        "split_files": split_files,
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
        "converted_frames": converted_frames,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-split", default="train_processed.txt")
    parser.add_argument("--val-split", default="val_processed.txt")
    parser.add_argument(
        "--test-split",
        default=None,
        help="Optional real test split; test_min_processed is intentionally not used by default",
    )
    parser.add_argument("--motion-path", default="motions")
    parser.add_argument("--artifact-path", default="artifacts")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    split_files = {"train": args.train_split, "val": args.val_split}
    if args.test_split:
        split_files["test"] = args.test_split
    summary = build_dataset(
        Path(args.source_root),
        Path(args.output),
        split_files=split_files,
        motion_path=args.motion_path,
        artifact_path=args.artifact_path,
        workers=args.workers,
        fps=args.fps,
        min_frames=args.min_frames,
        skip_missing=args.skip_missing,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
