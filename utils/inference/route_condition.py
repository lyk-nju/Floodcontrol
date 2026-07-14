"""Model-independent user route timeline state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from utils.inference.geometry import sample_plan_by_time


class RouteReferenceMode(str, Enum):
    """How a world-space route will be interpreted by a future compiler."""

    ABSOLUTE = "absolute"
    RELATIVE_TO_ACTOR = "relative_to_actor"
    GOAL_POINT = "goal_point"
    SPARSE = "sparse"


@dataclass
class RoutePlan:
    """World-space user route."""

    times: np.ndarray
    points_xyz: np.ndarray
    start_commit_index: int
    version: int
    source: str


@dataclass
class RouteUpdate:
    """Pending mid-session route update."""

    old_route: RoutePlan | None
    new_route: RoutePlan
    edit_commit_index: int
    effective_commit_index: int
    delay_tokens: int
    blend_tokens: int
    version: int


class RouteConditionState:
    """Manage the current user route and pending route edits."""

    def __init__(
        self,
        route: RoutePlan | None = None,
        *,
        mode: RouteReferenceMode | str = RouteReferenceMode.ABSOLUTE,
    ):
        self.route = route
        self.pending_update: RouteUpdate | None = None
        self.mode = RouteReferenceMode(mode)

    def set_mode(self, mode: RouteReferenceMode | str) -> None:
        self.mode = RouteReferenceMode(mode)

    def clear(self) -> None:
        self.route = None
        self.pending_update = None

    def update_route(
        self,
        route: RoutePlan,
        *,
        edit_commit_idx: int,
        delay_tokens: int = 0,
        blend_tokens: int = 0,
    ) -> RouteUpdate:
        effective_commit = int(edit_commit_idx) + max(0, int(delay_tokens))
        update = RouteUpdate(
            old_route=self.route,
            new_route=route,
            edit_commit_index=int(edit_commit_idx),
            effective_commit_index=effective_commit,
            delay_tokens=max(0, int(delay_tokens)),
            blend_tokens=max(0, int(blend_tokens)),
            version=int(route.version),
        )
        self.pending_update = update
        if update.delay_tokens == 0:
            self.route = route
            self.pending_update = None
        return update

    def active_route(self, commit_idx: int) -> RoutePlan | None:
        if (
            self.pending_update is not None
            and int(commit_idx) >= self.pending_update.effective_commit_index
        ):
            self.route = self.pending_update.new_route
            self.pending_update = None
        return self.route

def sample_route_future(
    route: RoutePlan,
    *,
    current_commit: int,
    current_root_xyz: np.ndarray,
    horizon_tokens: int,
    token_dt: float,
    reanchor_to_current_root: bool,
) -> np.ndarray:
    """Sample a route's future positions in world space."""
    elapsed_tokens = max(0, int(current_commit) - int(route.start_commit_index))
    query_times = (
        float(elapsed_tokens) * float(token_dt)
        + np.arange(int(horizon_tokens), dtype=np.float32) * float(token_dt)
    )
    future = sample_plan_by_time(route.times, route.points_xyz, query_times)
    if reanchor_to_current_root and len(future) > 0:
        root = np.asarray(current_root_xyz, dtype=np.float32).reshape(3)
        anchor = sample_plan_by_time(
            route.times,
            route.points_xyz,
            np.asarray([query_times[0]], dtype=np.float32),
        )[0]
        future = root[None, :] + (future - anchor[None, :])
    return future.astype(np.float32)


def reanchor_route_to_xz(route: RoutePlan, anchor_xz) -> RoutePlan:
    """Translate a route so its local t=0 point matches ``anchor_xz``."""
    points = np.asarray(route.points_xyz, dtype=np.float32)
    if points.size == 0:
        return route
    anchor = np.asarray(anchor_xz, dtype=np.float32).reshape(2)
    route_zero = sample_plan_by_time(
        np.asarray(route.times, dtype=np.float32),
        points,
        np.asarray([0.0], dtype=np.float32),
    )[0]
    offset = anchor - route_zero[[0, 2]]
    if float(np.linalg.norm(offset)) <= 1e-7:
        return route
    shifted = points.copy()
    shifted[:, [0, 2]] += offset[None, :]
    return RoutePlan(
        times=np.asarray(route.times, dtype=np.float32).copy(),
        points_xyz=shifted.astype(np.float32),
        start_commit_index=int(route.start_commit_index),
        version=int(route.version),
        source=str(route.source),
    )


__all__ = [
    "RouteConditionState",
    "RoutePlan",
    "RouteReferenceMode",
    "RouteUpdate",
    "reanchor_route_to_xz",
    "sample_route_future",
]
