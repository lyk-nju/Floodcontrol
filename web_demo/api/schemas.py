"""Strict JSON request schemas for the Web API boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.inference import GuidanceConfig, RouteEndBehavior, RouteReference
from utils.inference.geometry import validate_route_points


def _mapping(payload) -> dict:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _boolean(value, *, name: str, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean")


def _finite_float(value, *, name: str, default=None):
    if value is None:
        return default
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _guidance(payload) -> GuidanceConfig | None:
    if payload is None:
        return None
    values = _mapping(payload)
    return GuidanceConfig(
        mode=str(values.get("mode", "separated")),
        scale_text=_finite_float(
            values.get("scale_text", 1.0), name="scale_text"
        ),
        scale_constraint=_finite_float(
            values.get("scale_constraint", 1.0), name="scale_constraint"
        ),
        scale_joint=_finite_float(
            values.get("scale_joint", 1.0), name="scale_joint"
        ),
    )


@dataclass(frozen=True)
class StartSessionRequest:
    text: str
    seed: int
    initial_world_xz: tuple[float, float]
    initial_yaw: float | None
    force: bool
    guidance: GuidanceConfig | None
    route: "UpdateRouteRequest | None"

    @classmethod
    def from_payload(cls, payload) -> "StartSessionRequest":
        values = _mapping(payload)
        xz = np.asarray(values.get("initial_world_xz", [0.0, 0.0]), dtype=np.float32)
        if tuple(xz.shape) != (2,) or not bool(np.isfinite(xz).all()):
            raise ValueError("initial_world_xz must be a finite [2] value")
        try:
            seed = int(values.get("seed", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("seed must be an integer") from exc
        return cls(
            text=str(values.get("text", "")),
            seed=seed,
            initial_world_xz=(float(xz[0]), float(xz[1])),
            initial_yaw=_finite_float(
                values.get("initial_yaw"), name="initial_yaw", default=None
            ),
            force=_boolean(values.get("force"), name="force", default=False),
            guidance=_guidance(values.get("guidance")),
            route=(
                None
                if values.get("route") is None
                else UpdateRouteRequest.from_payload(values["route"])
            ),
        )


@dataclass(frozen=True)
class UpdateTextRequest:
    text: str

    @classmethod
    def from_payload(cls, payload) -> "UpdateTextRequest":
        values = _mapping(payload)
        if "text" not in values:
            raise ValueError("text is required")
        return cls(text=str(values["text"]))


@dataclass(frozen=True)
class UpdateRouteRequest:
    points_xz: np.ndarray
    duration_seconds: float
    reference: RouteReference
    end_behavior: RouteEndBehavior
    source: str

    @classmethod
    def from_payload(cls, payload) -> "UpdateRouteRequest":
        values = _mapping(payload)
        if "points_xz" not in values:
            raise ValueError("points_xz is required")
        points = validate_route_points(values["points_xz"])
        duration = _finite_float(
            values.get("duration_seconds", 4.0), name="duration_seconds"
        )
        if duration is None:
            raise ValueError("duration_seconds must be a finite number")
        if duration < 0 or (len(points) > 1 and duration <= 0):
            raise ValueError(
                "duration_seconds must be positive for a multi-point route"
            )
        return cls(
            points_xz=points,
            duration_seconds=duration,
            reference=RouteReference(
                values.get("reference", RouteReference.RELATIVE_TO_ACTOR.value)
            ),
            end_behavior=RouteEndBehavior(
                values.get("end_behavior", RouteEndBehavior.HOLD.value)
            ),
            source=str(values.get("source", "manual")),
        )


@dataclass(frozen=True)
class UpdateGuidanceRequest:
    guidance: GuidanceConfig

    @classmethod
    def from_payload(cls, payload) -> "UpdateGuidanceRequest":
        guidance = _guidance(_mapping(payload))
        assert guidance is not None
        return cls(guidance=guidance)


__all__ = [
    "StartSessionRequest",
    "UpdateGuidanceRequest",
    "UpdateRouteRequest",
    "UpdateTextRequest",
]
