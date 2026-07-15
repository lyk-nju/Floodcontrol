"""Strict, stateless geometry helpers for world-space XZ routes."""

from __future__ import annotations

import numpy as np


def validate_route_points(points_xz: np.ndarray) -> np.ndarray:
    """Return an owned float32 ``[N,2]`` route array.

    Route geometry is planar by definition. Root height and heading are typed
    root observations and must not be hidden in an XYZ waypoint convention.
    """

    points = np.asarray(points_xz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xz must have shape [N,2], got {points.shape}")
    if len(points) == 0:
        raise ValueError("a route must contain at least one point")
    if not bool(np.isfinite(points).all()):
        raise ValueError("points_xz must contain only finite values")
    return points.copy()


def validate_route_times(times: np.ndarray, *, point_count: int) -> np.ndarray:
    """Return strictly increasing float32 timestamps starting at zero."""

    values = np.asarray(times, dtype=np.float32).reshape(-1)
    if len(values) != int(point_count):
        raise ValueError(
            f"times/points length mismatch: {len(values)} != {point_count}"
        )
    if not bool(np.isfinite(values).all()):
        raise ValueError("route times must contain only finite values")
    if len(values) == 0:
        raise ValueError("a route must contain at least one timestamp")
    if abs(float(values[0])) > 1e-6:
        raise ValueError("route times must start at zero")
    if len(values) > 1 and bool((np.diff(values) <= 0).any()):
        raise ValueError("route times must be strictly increasing")
    return values.copy()


def project_to_route(
    point_xz: np.ndarray,
    route_xz: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Project one XZ point onto a route.

    Returns the projected point, segment index, and segment interpolation
    parameter. A single-point route returns that point with ``(0, 0)``.
    """

    point = np.asarray(point_xz, dtype=np.float32).reshape(-1)
    if tuple(point.shape) != (2,) or not bool(np.isfinite(point).all()):
        raise ValueError("point_xz must be a finite [2] value")
    route = validate_route_points(route_xz)
    if len(route) == 1:
        return route[0], 0, 0.0

    best_distance = float("inf")
    best_point = route[0]
    best_segment = 0
    best_parameter = 0.0
    for segment in range(len(route) - 1):
        start = route[segment]
        delta = route[segment + 1] - start
        length_squared = float(np.dot(delta, delta))
        parameter = (
            0.0
            if length_squared <= 1e-12
            else float(np.clip(np.dot(point - start, delta) / length_squared, 0, 1))
        )
        projected = start + parameter * delta
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_point = projected
            best_segment = segment
            best_parameter = parameter
    return best_point.astype(np.float32), best_segment, best_parameter


def sample_route_by_distance(
    points_xz: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Sample an XZ polyline at explicit non-negative arc distances."""

    points = validate_route_points(points_xz)
    query = np.asarray(distances, dtype=np.float32).reshape(-1)
    if not bool(np.isfinite(query).all()) or bool((query < 0).any()):
        raise ValueError("distances must be finite and non-negative")
    if len(query) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, len(query), axis=0)

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)]
    )
    if float(cumulative[-1]) <= 1e-8:
        return np.repeat(points[:1], len(query), axis=0)
    clamped = np.clip(query, 0.0, float(cumulative[-1]))
    return np.column_stack(
        [np.interp(clamped, cumulative, points[:, axis]) for axis in range(2)]
    ).astype(np.float32)


def assign_times_by_arclength(
    points_xz: np.ndarray,
    duration_seconds: float,
) -> np.ndarray:
    """Assign constant-speed timestamps to XZ waypoints."""

    points = validate_route_points(points_xz)
    duration = float(duration_seconds)
    if not np.isfinite(duration) or duration < 0:
        raise ValueError("duration_seconds must be finite and non-negative")
    if len(points) == 1:
        return np.zeros(1, dtype=np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(
        [np.zeros(1, dtype=np.float32), np.cumsum(lengths, dtype=np.float32)]
    )
    total = float(cumulative[-1])
    if total <= 1e-8:
        return np.linspace(0.0, duration, len(points), dtype=np.float32)
    return (cumulative / total * duration).astype(np.float32)


def sample_timed_route(
    times: np.ndarray,
    points_xz: np.ndarray,
    query_times: np.ndarray,
    *,
    hold_after_end: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a validated time-parameterized XZ route.

    Returns sampled points and a validity mask. Queries before route start are
    invalid. Queries after the end either hold the final point or become
    invalid according to ``hold_after_end``.
    """

    points = validate_route_points(points_xz)
    route_times = validate_route_times(times, point_count=len(points))
    query = np.asarray(query_times, dtype=np.float32).reshape(-1)
    if not bool(np.isfinite(query).all()):
        raise ValueError("query_times must contain only finite values")
    sampled = np.zeros((len(query), 2), dtype=np.float32)
    if len(query) == 0:
        return sampled, np.zeros(0, dtype=bool)

    before_start = query < route_times[0]
    after_end = query > route_times[-1]
    valid = ~before_start
    if not hold_after_end:
        valid &= ~after_end
    for axis in range(2):
        sampled[:, axis] = np.interp(
            query, route_times, points[:, axis]
        ).astype(np.float32)
    sampled[~valid] = 0
    return sampled, valid


def translate_route(points_xz: np.ndarray, translation_xz: np.ndarray) -> np.ndarray:
    """Translate an XZ route without changing its timing."""

    points = validate_route_points(points_xz)
    translation = np.asarray(translation_xz, dtype=np.float32).reshape(-1)
    if tuple(translation.shape) != (2,) or not bool(np.isfinite(translation).all()):
        raise ValueError("translation_xz must be a finite [2] value")
    return (points + translation[None]).astype(np.float32)


__all__ = [
    "assign_times_by_arclength",
    "project_to_route",
    "sample_route_by_distance",
    "sample_timed_route",
    "translate_route",
    "validate_route_points",
    "validate_route_times",
]
