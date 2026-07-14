"""Build root5/body265 artifacts from the processed HumanML3D dataset.

The builder reads ``new_joint_vecs`` and source split TXT files from one
HumanML3D root, then writes a separate, resumable ``HumanML3D_motion`` dataset.
Rotations are explicitly identified as HumanML3D IK-derived rotations rather
than native AMASS/SMPL pose parameters.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from utils.conditions.vae import CONTRACT_VERSION, FRAMES_PER_TOKEN
from utils.motion_representation import (
    HUMANML_DIM,
    HUMANML_SOURCE_REPRESENTATION,
    humanml263_to_root_body_motion,
)


SOURCE_REPRESENTATION = HUMANML_SOURCE_REPRESENTATION
DEFAULT_SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


def artifact_is_current(source: Path, target: Path) -> bool:
    if not target.is_file():
        return False
    try:
        with np.load(target, allow_pickle=False) as data:
            return (
                str(np.asarray(data["contract_version"]).item()) == CONTRACT_VERSION
                and str(np.asarray(data["source_representation"]).item())
                == SOURCE_REPRESENTATION
                and str(np.asarray(data["source_sha256"]).item()) == _sha256(source)
            )
    except (KeyError, OSError, ValueError):
        return False


def process_file(source: Path, target: Path, *, fps: float = 20.0) -> dict:
    feature = np.load(source, allow_pickle=False)
    if feature.ndim != 2 or feature.shape[-1] != HUMANML_DIM:
        raise ValueError(
            f"HumanML3D source must be [F,{HUMANML_DIM}], got {feature.shape} at {source}"
        )
    usable = feature.shape[0] // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN
    if usable < FRAMES_PER_TOKEN:
        raise ValueError(f"{source} has fewer than four usable frames")
    motion = torch.from_numpy(feature[:usable]).float()
    root, body, feature_valid = humanml263_to_root_body_motion(motion, fps=fps)
    source_sha256 = _sha256(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            root_motion=root.numpy(),
            body_motion=body.numpy(),
            body_feature_valid_mask=feature_valid.numpy(),
            contract_version=CONTRACT_VERSION,
            source_representation=SOURCE_REPRESENTATION,
            fps=np.float32(fps),
            source_sha256=source_sha256,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "frames": usable,
        "source_representation": SOURCE_REPRESENTATION,
        "source_sha256": source_sha256,
    }


def _convert_task(task: tuple[str, str, float]) -> tuple[str, int]:
    source_value, target_value, fps = task
    source, target = Path(source_value), Path(target_value)
    if artifact_is_current(source, target):
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


def _usable_frames(path: Path) -> int:
    feature = np.load(path, mmap_mode="r", allow_pickle=False)
    if feature.ndim != 2 or feature.shape[-1] != HUMANML_DIM:
        raise ValueError(
            f"HumanML3D source must be [F,{HUMANML_DIM}], got {feature.shape} at {path}"
        )
    return int(feature.shape[0]) // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN


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
    if not motion_root.is_dir():
        raise RuntimeError(
            f"HUMANML3D_DATA_REQUIRED: 263D source directory not found at {motion_root}"
        )
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
    if min_frames < FRAMES_PER_TOKEN or min_frames % FRAMES_PER_TOKEN:
        raise ValueError("min_frames must be a positive multiple of four")

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

    usable_frames: dict[str, int] = {}
    for names in source_split_names.values():
        for name in names:
            if name not in usable_frames:
                usable_frames[name] = _usable_frames(motion_root / f"{name}.npy")
    too_short_by_split = {
        split: [name for name in names if usable_frames[name] < min_frames]
        for split, names in source_split_names.items()
    }
    split_names = {
        split: [name for name in names if usable_frames[name] >= min_frames]
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
    for split, names in split_names.items():
        _atomic_write_text(
            output / f"{split}.txt", "".join(f"{name}\n" for name in names)
        )
    summary = {
        "contract_version": CONTRACT_VERSION,
        "source_representation": SOURCE_REPRESENTATION,
        "source_root": str(source_root.resolve()),
        "artifact_path": str(artifact_subdir),
        "fps": float(fps),
        "min_frames": min_frames,
        "splits": {name: len(values) for name, values in split_names.items()},
        "missing": {name: len(values) for name, values in missing_by_split.items()},
        "too_short": {name: len(values) for name, values in too_short_by_split.items()},
        "unique_artifacts": len(tasks),
        "converted": converted,
        "skipped": skipped,
        "converted_frames": total_frames,
    }
    _atomic_write_text(
        output / "build_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
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
