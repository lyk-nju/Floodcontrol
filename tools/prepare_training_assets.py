"""Prepare all offline assets required by Floodcontrol VAE and LDF training.

Expected sources match FloodDiffusion's processed releases::

    HumanML3D/{new_joint_vecs,texts,train.txt,val.txt,test.txt}
    BABEL_streamed/{motions,texts,train_processed.txt,val_processed.txt}

``pre-vae`` builds motion/statistics/text assets. ``post-vae`` computes latent
statistics with a trained EMA BodyVAE. ``all`` runs both and ``verify`` only
checks the resulting training contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

# Keep the file executable both as ``python -m tools.prepare_training_assets``
# and as ``python tools/prepare_training_assets.py`` from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from datasets.babel import BABELDataset
from datasets.humanml3d import HumanML3DDataset
from tools.build_motion_artifact import atomic_write_text
from utils.initialize import load_config
from utils.training.ldf.text import TextEmbeddingLookup


MOTION_STATISTIC_SHAPES = {
    "local_root_mean": (4,),
    "local_root_std": (4,),
    "body_cont_mean": (261,),
    "body_cont_std": (261,),
}
ROOT_STATISTIC_SHAPES = {"root_mean": (5,), "root_std": (5,)}


@dataclass(frozen=True)
class AssetPaths:
    raw_data: Path
    deps: Path
    humanml_source: Path
    babel_source: Path
    humanml: Path
    babel: Path
    humanml_stats: Path
    multi_stats: Path
    root_stats: Path
    humanml_t5: Path
    multi_t5: Path
    report: Path


class PreparationReport:
    """Atomic, human-readable record of completed or reused stages."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload = {
            "schema_version": 1,
            "raw_data_root": str(path.parent.resolve()),
            "stages": {},
        }
        if path.is_file():
            try:
                previous = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                previous = None
            if isinstance(previous, dict) and previous.get("schema_version") == 1:
                self.payload = previous

    def record(self, stage: str, status: str, details: Mapping[str, object]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.payload.setdefault("stages", {})[stage] = {
            "status": status,
            "details": dict(details),
            "updated_at": now,
        }
        self.payload["updated_at"] = now
        atomic_write_text(
            self.path,
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n",
        )


def _resolve_repo_file(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _resolve_paths(args) -> AssetPaths:
    raw_data = Path(args.raw_data_root).expanduser().resolve()
    deps = (
        Path(args.deps_root).expanduser().resolve()
        if args.deps_root
        else raw_data.parent / "deps"
    )
    humanml_source = (
        Path(args.humanml_source).expanduser().resolve()
        if args.humanml_source
        else raw_data / "HumanML3D"
    )
    babel_source = (
        Path(args.babel_source).expanduser().resolve()
        if args.babel_source
        else raw_data / "BABEL_streamed"
    )
    humanml = raw_data / "HumanML3D_motion"
    babel = raw_data / "BABEL_motion"
    return AssetPaths(
        raw_data=raw_data,
        deps=deps,
        humanml_source=humanml_source,
        babel_source=babel_source,
        humanml=humanml,
        babel=babel,
        humanml_stats=humanml / "motion_stats.npz",
        multi_stats=raw_data / "HumanML3D_BABEL_motion_stats.npz",
        root_stats=humanml / "root_stats.npz",
        humanml_t5=humanml / "t5_text_embeddings.pt",
        multi_t5=raw_data / "HumanML3D_BABEL_t5_text_embeddings.pt",
        report=raw_data / "training_assets.json",
    )


def _overrides(paths: AssetPaths) -> dict[str, str]:
    return {"dirs.raw_data": str(paths.raw_data), "dirs.deps": str(paths.deps)}


def _run(command: list[str], *, env: Mapping[str, str] | None = None) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=None if env is None else dict(env),
        check=True,
    )


def _stage(
    report: PreparationReport,
    name: str,
    *,
    validate: Callable[[], Mapping[str, object]],
    action: Callable[[], None],
    force: bool,
) -> Mapping[str, object]:
    if not force:
        try:
            details = dict(validate())
        except (FileNotFoundError, RuntimeError, ValueError):
            pass
        else:
            print(f"[{name}] reused", flush=True)
            report.record(name, "reused", details)
            return details
    print(f"[{name}] running", flush=True)
    action()
    details = dict(validate())
    report.record(name, "completed", details)
    print(f"[{name}] completed", flush=True)
    return details


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"split must contain unique sample ids: {path}")
    return names


def _validate_dataset(root: Path, splits: tuple[str, ...]) -> dict[str, object]:
    counts: dict[str, int] = {}
    split_union: set[str] = set()
    for split in splits:
        names = _read_split(root / f"{split}.txt")
        missing_motion = [
            name for name in names if not (root / "artifacts" / f"{name}.npz").is_file()
        ]
        missing_text = [
            name for name in names if not (root / "texts" / f"{name}.txt").is_file()
        ]
        if missing_motion or missing_text:
            raise RuntimeError(
                f"incomplete processed dataset at {root}: "
                f"motion={missing_motion[:5]}, text={missing_text[:5]}"
            )
        counts[split] = len(names)
        split_union.update(names)
    all_names = _read_split(root / "all.txt")
    if set(all_names) != split_union:
        raise RuntimeError(
            f"processed dataset all.txt does not equal the split union at {root}"
        )
    return {"path": str(root.resolve()), "splits": counts, "all": len(all_names)}


def _validate_statistics(
    path: Path,
    shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as values:
        fields = set(values.files)
        missing = set(shapes) - fields
        unexpected = fields - set(shapes) - {"metadata"}
        if missing or unexpected:
            raise ValueError(
                f"invalid statistics fields at {path}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        for name, shape in shapes.items():
            value = np.asarray(values[name])
            if tuple(value.shape) != shape or not np.isfinite(value).all():
                raise ValueError(f"invalid {name} at {path}")
            if name.endswith("std") and np.any(value <= 0):
                raise ValueError(f"{name} must be positive at {path}")
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "fields": {name: list(shape) for name, shape in shapes.items()},
    }


def _validate_t5(path: Path, config: str, paths: AssetPaths) -> dict[str, object]:
    cfg = load_config(config, _overrides(paths))
    table = TextEmbeddingLookup(
        path,
        expected_dim=int(cfg.model.params.text_dim),
        expected_text_len=int(cfg.model.params.text_len),
    )
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "captions": len(table),
        "content_id": table.content_id,
    }


def _pending_path(output: Path, suffix: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return output.with_name(f".{output.stem}.{os.getpid()}.pending{suffix}")


def _run_atomic(
    command: list[str],
    *,
    output: Path,
    pending: Path,
    validate: Callable[[Path], object],
    env: Mapping[str, str] | None = None,
) -> None:
    pending.unlink(missing_ok=True)
    try:
        _run(command, env=env)
        validate(pending)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)


def _require_sources(paths: AssetPaths) -> None:
    required = [
        paths.humanml_source / "new_joint_vecs",
        paths.humanml_source / "texts",
        paths.humanml_source / "train.txt",
        paths.humanml_source / "val.txt",
        paths.humanml_source / "test.txt",
        paths.babel_source / "motions",
        paths.babel_source / "texts",
        paths.babel_source / "train_processed.txt",
        paths.babel_source / "val_processed.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"SOURCE_DATA_REQUIRED: missing {missing}")


def _prepare_motion(args, paths: AssetPaths, report: PreparationReport) -> None:
    _require_sources(paths)
    python = sys.executable
    _stage(
        report,
        "humanml_motion",
        validate=lambda: _validate_dataset(paths.humanml, ("train", "val", "test")),
        action=lambda: _run(
            [
                python,
                "-m",
                "tools.preprocess_humanml3d",
                "--source-root",
                str(paths.humanml_source),
                "--output",
                str(paths.humanml),
                "--workers",
                str(args.workers),
                "--fps",
                str(args.fps),
                "--min-frames",
                str(args.min_frames),
            ]
        ),
        force=args.force,
    )
    _stage(
        report,
        "babel_motion",
        validate=lambda: _validate_dataset(paths.babel, ("train", "val")),
        action=lambda: _run(
            [
                python,
                "-m",
                "tools.preprocess_babel",
                "--source-root",
                str(paths.babel_source),
                "--output",
                str(paths.babel),
                "--workers",
                str(args.workers),
                "--fps",
                str(args.fps),
                "--min-frames",
                str(args.min_frames),
            ]
        ),
        force=args.force,
    )


def _statistics_command(config: str, paths: AssetPaths, output: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "tools.compute_vae_stats",
        "--config",
        config,
        "--override",
        f"dirs.raw_data={paths.raw_data}",
        f"dirs.deps={paths.deps}",
        "--output",
        str(output),
    ]


def _prepare_statistics(args, paths: AssetPaths, report: PreparationReport) -> None:
    for name, config, output in (
        ("humanml_motion_statistics", args.vae_config, paths.humanml_stats),
        ("multi_motion_statistics", args.vae_multi_config, paths.multi_stats),
    ):
        pending = _pending_path(output, ".npz")
        _stage(
            report,
            name,
            validate=lambda output=output: _validate_statistics(
                output, MOTION_STATISTIC_SHAPES
            ),
            action=lambda config=config, output=output, pending=pending: _run_atomic(
                _statistics_command(config, paths, pending),
                output=output,
                pending=pending,
                validate=lambda value: _validate_statistics(
                    value, MOTION_STATISTIC_SHAPES
                ),
            ),
            force=args.force,
        )

    cfg = load_config(args.ldf_config, _overrides(paths))
    pending = _pending_path(paths.root_stats, ".npz")
    root_command = [
        sys.executable,
        "-m",
        "tools.compute_ldf_root_stats",
        "--train-meta-paths",
        str(paths.humanml / "train.txt"),
        "--artifact-path",
        "artifacts",
        "--output",
        str(pending),
        "--fps",
        str(args.fps),
        "--min-frames",
        str(cfg.data.min_frames),
        "--max-frames",
        str(cfg.data.max_frames),
        "--windows-per-sample",
        str(args.root_windows_per_sample),
        "--active-tokens",
        str(cfg.model.params.chunk_size),
        "--seed",
        str(args.root_seed),
    ]
    _stage(
        report,
        "ldf_root_statistics",
        validate=lambda: _validate_statistics(paths.root_stats, ROOT_STATISTIC_SHAPES),
        action=lambda: _run_atomic(
            root_command,
            output=paths.root_stats,
            pending=pending,
            validate=lambda value: _validate_statistics(value, ROOT_STATISTIC_SHAPES),
        ),
        force=args.force,
    )


def _parse_devices(value: str) -> tuple[str, ...]:
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices or ("cpu" in devices and devices != ("cpu",)):
        raise ValueError("--t5-devices must be cpu or comma-separated CUDA ids")
    if devices != ("cpu",) and any(not value.isdigit() for value in devices):
        raise ValueError("--t5-devices contains a non-integer CUDA id")
    return devices


def _t5_command(
    *,
    config: str,
    output: Path,
    paths: AssetPaths,
    devices: tuple[str, ...],
    batch_size: int,
    reuse: bool,
) -> tuple[list[str], dict[str, str]]:
    tool = REPO_ROOT / "tools" / "pretokenize_t5_text.py"
    arguments = [
        str(tool),
        "--config",
        config,
        "--override",
        f"dirs.raw_data={paths.raw_data}",
        f"dirs.deps={paths.deps}",
        "--output",
        str(output),
        "--batch_size",
        str(batch_size),
    ]
    if reuse:
        arguments.append("--reuse-existing")
    environment = os.environ.copy()
    if devices == ("cpu",):
        return [sys.executable, *arguments, "--device", "cpu"], environment
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    if len(devices) == 1:
        return [sys.executable, *arguments, "--device", "cuda:0"], environment
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={len(devices)}",
        *arguments,
    ], environment


def _prepare_t5(args, paths: AssetPaths, report: PreparationReport) -> None:
    if args.skip_t5:
        print("[t5] skipped by --skip-t5", flush=True)
        return
    devices = _parse_devices(args.t5_devices)
    for config in (args.ldf_config, args.ldf_multi_config):
        cfg = load_config(config, _overrides(paths))
        checkpoint = Path(str(cfg.text_encoder.checkpoint_path))
        tokenizer = Path(str(cfg.text_encoder.tokenizer_path))
        if not checkpoint.is_file() or not tokenizer.is_dir():
            raise RuntimeError(
                f"T5_DEPENDENCY_REQUIRED: missing {checkpoint} or {tokenizer}"
            )

    human_pending = _pending_path(paths.humanml_t5, ".pt")

    def human_action() -> None:
        command, environment = _t5_command(
            config=args.ldf_config,
            output=human_pending,
            paths=paths,
            devices=devices,
            batch_size=args.t5_batch_size,
            reuse=False,
        )
        _run_atomic(
            command,
            output=paths.humanml_t5,
            pending=human_pending,
            validate=lambda value: _validate_t5(value, args.ldf_config, paths),
            env=environment,
        )

    _stage(
        report,
        "humanml_t5",
        validate=lambda: _validate_t5(paths.humanml_t5, args.ldf_config, paths),
        action=human_action,
        force=args.force,
    )

    multi_pending = _pending_path(paths.multi_t5, ".pt")

    def multi_action() -> None:
        multi_pending.unlink(missing_ok=True)
        shutil.copy2(paths.humanml_t5, multi_pending)
        command, environment = _t5_command(
            config=args.ldf_multi_config,
            output=multi_pending,
            paths=paths,
            devices=devices,
            batch_size=args.t5_batch_size,
            reuse=True,
        )
        try:
            _run(command, env=environment)
            _validate_t5(multi_pending, args.ldf_multi_config, paths)
            multi_pending.replace(paths.multi_t5)
        finally:
            multi_pending.unlink(missing_ok=True)

    _stage(
        report,
        "multi_t5",
        validate=lambda: _validate_t5(paths.multi_t5, args.ldf_multi_config, paths),
        action=multi_action,
        force=args.force,
    )


def _prepare_latent(args, paths: AssetPaths, report: PreparationReport) -> Path:
    if not args.vae_checkpoint:
        raise RuntimeError("VAE_CHECKPOINT_REQUIRED: set --vae-checkpoint")
    checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = (
        Path(args.latent_output).expanduser().resolve()
        if args.latent_output
        else checkpoint.parent / "latent_stats.npz"
    )
    latent_dim = int(load_config(args.vae_config, _overrides(paths)).model.params.latent_dim)
    shapes = {"mean": (latent_dim,), "std": (latent_dim,)}
    pending = _pending_path(output, ".npz")
    command = [
        sys.executable,
        "-m",
        "tools.compute_vae_latent_stats",
        "--train-meta-paths",
        str(paths.humanml / "train.txt"),
        "--artifact-path",
        "artifacts",
        "--config",
        args.vae_config,
        "--checkpoint",
        str(checkpoint),
        "--motion-stats",
        str(paths.humanml_stats),
        "--output",
        str(pending),
        "--device",
        args.latent_device,
        "--yaw-seed",
        str(args.latent_yaw_seed),
        "--batch-size",
        str(args.latent_batch_size),
    ]
    _stage(
        report,
        "vae_latent_statistics",
        validate=lambda: _validate_statistics(output, shapes),
        action=lambda: _run_atomic(
            command,
            output=output,
            pending=pending,
            validate=lambda value: _validate_statistics(value, shapes),
        ),
        force=args.force,
    )
    return output


def _verify(
    args,
    paths: AssetPaths,
    *,
    latent_stats: Path | None,
) -> dict[str, object]:
    details: dict[str, object] = {
        "humanml": _validate_dataset(paths.humanml, ("train", "val", "test")),
        "babel": _validate_dataset(paths.babel, ("train", "val")),
        "humanml_motion_stats": _validate_statistics(
            paths.humanml_stats, MOTION_STATISTIC_SHAPES
        ),
        "multi_motion_stats": _validate_statistics(
            paths.multi_stats, MOTION_STATISTIC_SHAPES
        ),
        "root_stats": _validate_statistics(paths.root_stats, ROOT_STATISTIC_SHAPES),
    }
    if not args.skip_t5:
        details["humanml_t5"] = _validate_t5(paths.humanml_t5, args.ldf_config, paths)
        details["multi_t5"] = _validate_t5(paths.multi_t5, args.ldf_multi_config, paths)

    human = HumanML3DDataset(
        meta_paths=[paths.humanml / "train.txt"],
        split="train",
        artifact_path="artifacts",
        text_path="texts",
        fps=args.fps,
    )
    babel = BABELDataset(
        meta_paths=[paths.babel / "train.txt"],
        split="train",
        artifact_path="artifacts",
        text_path="texts",
        fps=args.fps,
    )
    for dataset in (human, babel):
        sample = dataset[0]
        if sample["root_motion"].shape[-1] != 5 or sample["body_motion"].shape[-1] != 265:
            raise ValueError("processed sample does not satisfy root5/body265")

    if latent_stats is not None:
        latent_dim = int(load_config(args.vae_config, _overrides(paths)).model.params.latent_dim)
        details["latent_stats"] = _validate_statistics(
            latent_stats, {"mean": (latent_dim,), "std": (latent_dim,)}
        )
        checkpoint = Path(args.vae_checkpoint).expanduser().resolve()
        if args.skip_t5:
            details["ldf_configs"] = "not checked because --skip-t5 was set"
        else:
            from train_ldf import _validate_training_config

            for config in (args.ldf_config, args.ldf_multi_config):
                cfg = load_config(
                    config,
                    {
                        **_overrides(paths),
                        "vae.checkpoint_path": str(checkpoint),
                        "vae.params.latent_stats_path": str(latent_stats),
                    },
                )
                _validate_training_config(cfg)
            details["ldf_configs"] = [args.ldf_config, args.ldf_multi_config]
    return details


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pre-vae", "post-vae", "all", "verify"))
    parser.add_argument("--raw-data-root", required=True)
    parser.add_argument("--deps-root")
    parser.add_argument("--humanml-source")
    parser.add_argument("--babel-source")
    parser.add_argument("--vae-config", default="configs/vae.yaml")
    parser.add_argument("--vae-multi-config", default="configs/vae_multi.yaml")
    parser.add_argument("--ldf-config", default="configs/ldf.yaml")
    parser.add_argument("--ldf-multi-config", default="configs/ldf_multi.yaml")
    parser.add_argument("--vae-checkpoint")
    parser.add_argument("--latent-output")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--min-frames", type=int, default=20)
    parser.add_argument("--root-windows-per-sample", type=int, default=1)
    parser.add_argument("--root-seed", type=int, default=1234)
    parser.add_argument("--t5-devices", default="0")
    parser.add_argument("--t5-batch-size", type=int, default=4)
    parser.add_argument("--skip-t5", action="store_true")
    parser.add_argument("--latent-device", default="cuda:0")
    parser.add_argument("--latent-batch-size", type=int, default=128)
    parser.add_argument("--latent-yaw-seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    for name in ("vae_config", "vae_multi_config", "ldf_config", "ldf_multi_config"):
        setattr(args, name, _resolve_repo_file(getattr(args, name)))
    for name in (
        "workers",
        "min_frames",
        "root_windows_per_sample",
        "t5_batch_size",
        "latent_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if float(args.fps) != 20.0:
        raise ValueError("Floodcontrol HumanML3D/BABEL protocol requires 20 FPS")

    paths = _resolve_paths(args)
    paths.raw_data.mkdir(parents=True, exist_ok=True)
    report = PreparationReport(paths.report)
    latent_stats: Path | None = None

    if args.phase in {"pre-vae", "all"}:
        _prepare_motion(args, paths, report)
        _prepare_statistics(args, paths, report)
        _prepare_t5(args, paths, report)
        report.record("pre_vae_verification", "completed", _verify(args, paths, latent_stats=None))
    if args.phase in {"post-vae", "all"}:
        latent_stats = _prepare_latent(args, paths, report)
    if args.phase == "verify":
        if not args.vae_checkpoint:
            raise RuntimeError("VAE_CHECKPOINT_REQUIRED: verify checks LDF readiness")
        latent_stats = (
            Path(args.latent_output).expanduser().resolve()
            if args.latent_output
            else Path(args.vae_checkpoint).expanduser().resolve().parent
            / "latent_stats.npz"
        )
    if args.phase in {"post-vae", "all", "verify"}:
        report.record(
            "final_verification",
            "completed",
            _verify(args, paths, latent_stats=latent_stats),
        )
    print(f"training asset report: {paths.report}", flush=True)


if __name__ == "__main__":
    main()
