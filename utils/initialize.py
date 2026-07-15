"""Configuration, dynamic construction, and run-initialization helpers.

The project uses YAML targets such as ``models.vae_wan_1d.BodyVAE`` to build
models, Datasets, optimizers, and schedulers. This module owns that small
bootstrap layer plus the filesystem metadata created at the start of a run.
It does not own model, Dataset, or training-loop behavior.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf


DEFAULT_CONFIG_PATH = "configs/default.yaml"
PATH_CONFIG_CANDIDATES = (
    Path("configs/paths.yaml"),
    Path("configs/paths_default.yaml"),
)
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RUN_TIMESTAMP_ENV = "PL_RUN_TIME"


class ProjectConfig:
    """Merged project configuration with attribute and key access.

    Path settings are loaded first from ``configs/paths.yaml`` when present,
    otherwise from the tracked ``configs/paths_default.yaml``. The requested
    experiment YAML and explicit command-line overrides are then applied in
    that order.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = OmegaConf.create({})
        path_config = next(
            (path for path in PATH_CONFIG_CANDIDATES if path.is_file()),
            None,
        )
        if path_config is None:
            candidates = ", ".join(str(path) for path in PATH_CONFIG_CANDIDATES)
            raise FileNotFoundError(f"no path configuration found; checked {candidates}")
        self.merge_yaml(path_config)
        if config_path is not None:
            self.merge_yaml(config_path)
        if overrides:
            self.apply_overrides(overrides)

    def merge_yaml(self, config_path: str | Path) -> None:
        """Merge one YAML file into the current configuration."""
        loaded = OmegaConf.load(config_path)
        self.config = OmegaConf.merge(self.config, loaded)

    def apply_overrides(self, overrides: Mapping[str, Any]) -> None:
        """Apply dot-separated OmegaConf keys after parsing scalar strings."""
        for key, value in overrides.items():
            OmegaConf.update(self.config, key, _parse_override_value(value))

    def get(self, key: str, default: Any = None) -> Any:
        """Select a possibly nested key, returning ``default`` when absent."""
        return OmegaConf.select(self.config, key, default=default)

    def __getattr__(self, name: str) -> Any:
        return self.config[name]

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def save(self, path: str | Path) -> None:
        """Write the fully merged configuration to YAML."""
        OmegaConf.save(self.config, path)


def _parse_override_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
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


def parse_config_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the common project configuration CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Experiment YAML path")
    parser.add_argument(
        "--override",
        nargs="+",
        metavar="KEY=VALUE",
        help="Dot-separated OmegaConf overrides",
    )
    return parser.parse_args(argv)


def _parse_override_arguments(values: Sequence[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"override must use KEY=VALUE syntax, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override key must not be empty: {item!r}")
        overrides[key] = value.strip()
    return overrides


def load_config(
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ProjectConfig:
    """Load path settings, one experiment YAML, and optional overrides."""
    if config_path is None:
        arguments = parse_config_arguments()
        config_path = arguments.config
        overrides = _parse_override_arguments(arguments.override)
    return ProjectConfig(config_path, overrides)


def _resolve_target(target: str) -> Any:
    if not isinstance(target, str) or "." not in target:
        raise ValueError(f"target must be a fully-qualified name, got {target!r}")
    module_name, attribute_name = target.rsplit(".", 1)
    module = import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise AttributeError(f"target {target!r} does not exist") from error


def instantiate_target(
    target: str,
    *,
    cfg: Any = None,
    hfstyle: bool = False,
    **arguments: Any,
) -> Any:
    """Instantiate a fully-qualified class or factory from configuration."""
    constructor = _resolve_target(target)
    if not callable(constructor):
        raise TypeError(f"target {target!r} is not callable")
    if cfg is None:
        return constructor(**arguments)
    if hfstyle:
        cfg = constructor.config_class(config_obj=cfg)
    return constructor(cfg, **arguments)


def resolve_function(target: str) -> Callable[..., Any]:
    """Resolve and validate a fully-qualified function-like target."""
    function = _resolve_target(target)
    if not callable(function):
        raise TypeError(f"target {target!r} is not callable")
    return function


def save_run_snapshot(config: ProjectConfig, run_directory: str | Path) -> None:
    """Save the resolved YAML and Python sources used to start a run."""
    snapshot_directory = Path(run_directory) / "sanity_check"
    snapshot_directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config.config, snapshot_directory / f"{config.exp_name}.yaml")

    source_root = Path.cwd()
    excluded_directories = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "output_eval",
        "outputs",
        "sanity_check",
    }
    for root, directory_names, file_names in os.walk(source_root, topdown=True):
        directory_names[:] = [
            name for name in directory_names if name not in excluded_directories
        ]
        root_path = Path(root)
        for file_name in file_names:
            if not file_name.endswith(".py"):
                continue
            source = root_path / file_name
            destination = snapshot_directory / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def log_model_parameters(model: torch.nn.Module) -> None:
    """Log total, trainable, and frozen parameter counts on global rank zero."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    rank_zero_info(f"Total parameters: {total:,}")
    rank_zero_info(f"Trainable parameters: {trainable:,}")
    rank_zero_info(f"Non-trainable parameters: {total - trainable:,}")


def log_state_dict_summary(
    state_dict: Mapping[str, Any],
    named_parameters,
    named_buffers,
) -> None:
    """Log missing or unexpected state keys on global rank zero."""
    state_keys = set(state_dict)
    parameter_keys = {name for name, _ in named_parameters}
    buffer_keys = {name for name, _ in named_buffers}
    expected_keys = parameter_keys | buffer_keys

    unexpected = sorted(state_keys - expected_keys)
    missing_parameters = sorted(parameter_keys - state_keys)
    missing_buffers = sorted(buffer_keys - state_keys)
    if unexpected:
        rank_zero_info(f"Unexpected state_dict keys: {unexpected}")
    if missing_parameters:
        rank_zero_info(f"Missing parameter keys: {missing_parameters}")
    if missing_buffers:
        rank_zero_info(f"Missing buffer keys: {missing_buffers}")
    if not unexpected and not missing_parameters and not missing_buffers:
        rank_zero_info("All state_dict keys match model parameters and buffers")

    rank_zero_info(f"Total state_dict items: {len(state_keys)}")
    rank_zero_info(f"Total named parameters: {len(parameter_keys)}")
    rank_zero_info(f"Total named buffers: {len(buffer_keys)}")


def _resolve_global_rank() -> int:
    for key in ("GLOBAL_RANK", "RANK", "SLURM_PROCID", "LOCAL_RANK"):
        try:
            return int(os.environ[key])
        except (KeyError, ValueError):
            continue
    return 0


def get_shared_run_timestamp(
    base_directory: str | Path,
    *,
    environment_key: str = RUN_TIMESTAMP_ENV,
) -> str:
    """Return one run timestamp shared by every distributed process.

    Initialized torch.distributed jobs broadcast directly from rank zero. Jobs
    launched as independent processes coordinate through a short-lived file in
    the output directory. The value is cached in ``environment_key``.
    """
    cached = os.environ.get(environment_key)
    if cached:
        return cached

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        timestamp = (
            datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
            if torch.distributed.get_rank() == 0
            else None
        )
        values = [timestamp]
        torch.distributed.broadcast_object_list(values, src=0)
        timestamp = values[0]
        if timestamp is None:
            raise RuntimeError("failed to synchronize the run timestamp")
        os.environ[environment_key] = timestamp
        return timestamp

    base_directory = Path(base_directory)
    sync_directory = base_directory / ".run_time_sync"
    sync_directory.mkdir(parents=True, exist_ok=True)
    job_identity = (
        os.environ.get("SLURM_JOB_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("JOB_ID")
        or "default"
    )
    sync_file = sync_directory / f"{job_identity}.txt"

    if _resolve_global_rank() == 0:
        sync_file.unlink(missing_ok=True)
        timestamp = datetime.now().strftime(RUN_TIMESTAMP_FORMAT)
        sync_file.write_text(timestamp, encoding="utf-8")
    else:
        deadline = time.monotonic() + 1200.0
        while True:
            try:
                timestamp = sync_file.read_text(encoding="utf-8").strip()
                parsed = datetime.strptime(timestamp, RUN_TIMESTAMP_FORMAT)
                if abs((datetime.now() - parsed).total_seconds()) < 60:
                    break
            except (OSError, ValueError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("timed out waiting for the rank-zero run timestamp")
            time.sleep(0.1)

    os.environ[environment_key] = timestamp
    return timestamp


__all__ = [
    "ProjectConfig",
    "get_shared_run_timestamp",
    "instantiate_target",
    "load_config",
    "log_model_parameters",
    "log_state_dict_summary",
    "parse_config_arguments",
    "resolve_function",
    "save_run_snapshot",
]
