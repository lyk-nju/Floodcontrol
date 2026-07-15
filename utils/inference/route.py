"""Typed world-space route input for streaming inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from utils.inference.geometry import (
    sample_timed_route,
    translate_route,
    validate_route_points,
    validate_route_times,
)
from utils.token_frame import FRAMES_PER_TOKEN


class RouteReference(str, Enum):
    """Coordinate reference used only while accepting a route update."""

    WORLD = "world"
    RELATIVE_TO_ACTOR = "relative_to_actor"


class RouteEndBehavior(str, Enum):
    """Whether the final route point remains constrained after route end."""

    HOLD = "hold"
    RELEASE = "release"


@dataclass(frozen=True)
class RoutePlan:
    """A time-parameterized XZ route beginning at an absolute token index."""

    times: np.ndarray
    points_xz: np.ndarray
    start_token: int
    end_behavior: RouteEndBehavior = RouteEndBehavior.HOLD
    version: int = 0
    source: str = "manual"

    def __post_init__(self) -> None:
        points = validate_route_points(self.points_xz)
        times = validate_route_times(self.times, point_count=len(points))
        if int(self.start_token) < 0:
            raise ValueError("start_token must be non-negative")
        points.setflags(write=False)
        times.setflags(write=False)
        object.__setattr__(self, "points_xz", points)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "start_token", int(self.start_token))
        object.__setattr__(self, "end_behavior", RouteEndBehavior(self.end_behavior))
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "source", str(self.source))

    def resolve_world(
        self,
        reference: RouteReference | str,
        actor_world_xz: np.ndarray,
    ) -> "RoutePlan":
        """Resolve a relative input exactly once and return a world route."""

        mode = RouteReference(reference)
        points = self.points_xz
        if mode is RouteReference.RELATIVE_TO_ACTOR:
            points = translate_route(points, actor_world_xz)
        return RoutePlan(
            times=self.times,
            points_xz=points,
            start_token=self.start_token,
            end_behavior=self.end_behavior,
            version=self.version,
            source=self.source,
        )

    def sample_frames(
        self,
        frame_indices: np.ndarray,
        *,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample world XZ at absolute session frame indices."""

        rate = float(fps)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("fps must be finite and positive")
        frames = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if bool((frames < 0).any()):
            raise ValueError("frame_indices must be non-negative")
        start_frame = self.start_token * FRAMES_PER_TOKEN
        query_times = (frames.astype(np.float32) - float(start_frame)) / rate
        return sample_timed_route(
            self.times,
            self.points_xz,
            query_times,
            hold_after_end=self.end_behavior is RouteEndBehavior.HOLD,
        )


__all__ = ["RouteEndBehavior", "RoutePlan", "RouteReference"]
