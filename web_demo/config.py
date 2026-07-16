"""Typed configuration loading for the Floodcontrol Web runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from utils.inference import GuidanceConfig, InferenceConfig


@dataclass(frozen=True)
class WebConfig:
    """Server-owned runtime settings; route updates cannot mutate these values."""

    status: str
    message: str
    inference: InferenceConfig
    guidance: GuidanceConfig
    target_fps: float = 20.0
    buffer_target_chunks: int = 2
    buffer_capacity_chunks: int = 8
    chunk_wait_timeout_seconds: float = 0.5
    consumption_timeout_seconds: float = 5.0
    monitor_interval_seconds: float = 1.0
    worker_stop_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if float(self.target_fps) <= 0:
            raise ValueError("target_fps must be positive")
        if int(self.buffer_target_chunks) <= 0:
            raise ValueError("buffer_target_chunks must be positive")
        if int(self.buffer_capacity_chunks) < int(self.buffer_target_chunks):
            raise ValueError("buffer_capacity_chunks must be >= buffer_target_chunks")
        for name in (
            "chunk_wait_timeout_seconds",
            "consumption_timeout_seconds",
            "monitor_interval_seconds",
            "worker_stop_timeout_seconds",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    def template_defaults(self) -> dict:
        return {
            "target_fps": self.target_fps,
            "route_reference": "relative_to_actor",
            "route_end_behavior": "hold",
            "guidance": {
                "mode": self.guidance.mode,
                "scale_text": self.guidance.scale_text,
                "scale_constraint": self.guidance.scale_constraint,
                "scale_joint": self.guidance.scale_joint,
            },
        }


def _section(mapping: dict, name: str) -> dict:
    value = mapping.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def load_web_config(path: str | Path) -> WebConfig:
    """Load the standalone Web protocol section from ``configs/stream.yaml``."""

    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Web config does not exist: {config_path}")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Web config root must be a mapping")
    web = _section(raw, "web")
    inference_values = _section(web, "inference")
    guidance_values = _section(web, "guidance")
    buffer_values = _section(web, "buffer")
    session_values = _section(web, "session")
    return WebConfig(
        status=str(raw.get("status", "BLOCKED_ON_LDF_CHECKPOINT")),
        message=str(raw.get("message", "")),
        inference=InferenceConfig(
            window_tokens=int(inference_values.get("window_tokens", 50)),
            max_horizon_token=int(inference_values.get("max_horizon_token", 10)),
            num_denoise_steps=inference_values.get("num_denoise_steps", 10),
            rebase_on_roll=bool(inference_values.get("rebase_on_roll", True)),
        ),
        guidance=GuidanceConfig(
            mode=str(guidance_values.get("mode", "separated")),
            scale_text=float(guidance_values.get("scale_text", 1.0)),
            scale_constraint=float(
                guidance_values.get("scale_constraint", 1.0)
            ),
            scale_joint=float(guidance_values.get("scale_joint", 1.0)),
        ),
        target_fps=float(web.get("target_fps", 20.0)),
        buffer_target_chunks=int(buffer_values.get("target_chunks", 2)),
        buffer_capacity_chunks=int(buffer_values.get("capacity_chunks", 8)),
        chunk_wait_timeout_seconds=float(
            buffer_values.get("chunk_wait_timeout_seconds", 0.5)
        ),
        consumption_timeout_seconds=float(
            session_values.get("consumption_timeout_seconds", 5.0)
        ),
        monitor_interval_seconds=float(
            session_values.get("monitor_interval_seconds", 1.0)
        ),
        worker_stop_timeout_seconds=float(
            session_values.get("worker_stop_timeout_seconds", 5.0)
        ),
    )


__all__ = ["WebConfig", "load_web_config"]
