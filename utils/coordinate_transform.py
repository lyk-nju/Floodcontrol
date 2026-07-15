"""Coordinate transforms on the Y-up XZ ground plane.

This module owns only representation-independent geometry.  It operates on
angles, XZ points, XZ vectors and rotation matrices; it does not know about
root5, body265, tokens, model conditions or streaming state.

Axis convention:
    - +Y is up.
    - yaw rotates around +Y.
    - yaw=0 faces +Z.
    - ``heading_to_direction(yaw) = [sin(yaw), cos(yaw)]`` in XZ order.

Anchor tensors describe prefix dimensions.  For example, points shaped
``[B,F,2]`` may use an anchor position shaped ``[B,2]`` and yaw shaped ``[B]``;
the anchor is broadcast across the frame dimension.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


_PI = math.pi
_TWO_PI = 2.0 * math.pi


def _require_xz(name: str, value: Tensor) -> None:
    if not torch.is_tensor(value) or value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError(f"{name} must be a tensor ending in two XZ coordinates")


def _expand_yaw(yaw: Tensor, target_leading_dims: int) -> Tensor:
    if not torch.is_tensor(yaw):
        raise TypeError("yaw must be a torch.Tensor")
    if yaw.ndim > target_leading_dims:
        raise ValueError("yaw has more dimensions than the transformed values")
    while yaw.ndim < target_leading_dims:
        yaw = yaw.unsqueeze(-1)
    return yaw


def _expand_origin(origin_xz: Tensor, target_ndim: int) -> Tensor:
    _require_xz("origin_xz", origin_xz)
    if origin_xz.ndim > target_ndim:
        raise ValueError("origin_xz has more dimensions than the transformed points")
    while origin_xz.ndim < target_ndim:
        origin_xz = origin_xz.unsqueeze(-2)
    return origin_xz


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap angles in radians to the half-open interval ``[-pi, pi)``."""
    return (angle + _PI) % _TWO_PI - _PI


def yaw_to_matrix(yaw: Tensor) -> Tensor:
    """Convert physical yaw to a Y-up rotation matrix.

    Args:
        yaw: Angles in radians with arbitrary leading dimensions.

    Returns:
        Rotation matrices shaped ``yaw.shape + (3,3)``.
    """
    if not torch.is_tensor(yaw):
        raise TypeError("yaw must be a torch.Tensor")
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    zero = torch.zeros_like(cosine)
    one = torch.ones_like(cosine)
    return torch.stack(
        [
            torch.stack([cosine, zero, sine], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sine, zero, cosine], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_yaw(rotation: Tensor) -> Tensor:
    """Recover physical yaw from Y-up rotation matrices.

    Non-finite sine/cosine entries produce yaw zero rather than propagating a
    NaN into runtime coordinate state.
    """
    if not torch.is_tensor(rotation) or tuple(rotation.shape[-2:]) != (3, 3):
        raise ValueError("rotation must be a tensor ending in [3,3]")
    sine = rotation[..., 0, 2]
    cosine = rotation[..., 2, 2]
    yaw = torch.atan2(sine, cosine)
    finite = torch.isfinite(sine) & torch.isfinite(cosine)
    return wrap_angle(torch.where(finite, yaw, torch.zeros_like(yaw)))


def heading_to_direction(yaw: Tensor) -> Tensor:
    """Convert physical yaw to unit forward directions in XZ order."""
    if not torch.is_tensor(yaw):
        raise TypeError("yaw must be a torch.Tensor")
    return torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)


def rotate_vectors_world_to_local(vectors_world: Tensor, frame_yaw: Tensor) -> Tensor:
    """Rotate world-space XZ vectors into a local heading frame.

    Translation is intentionally absent because directions, velocities and
    displacements are vectors rather than points.
    """
    _require_xz("vectors_world", vectors_world)
    yaw = _expand_yaw(frame_yaw, vectors_world.ndim - 1)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x_world, z_world = vectors_world.unbind(dim=-1)
    return torch.stack(
        [
            cosine * x_world - sine * z_world,
            sine * x_world + cosine * z_world,
        ],
        dim=-1,
    )


def rotate_vectors_local_to_world(vectors_local: Tensor, frame_yaw: Tensor) -> Tensor:
    """Rotate local-frame XZ vectors into world coordinates."""
    _require_xz("vectors_local", vectors_local)
    yaw = _expand_yaw(frame_yaw, vectors_local.ndim - 1)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x_local, z_local = vectors_local.unbind(dim=-1)
    return torch.stack(
        [
            cosine * x_local + sine * z_local,
            -sine * x_local + cosine * z_local,
        ],
        dim=-1,
    )


def transform_points_world_to_local(
    points_world: Tensor,
    origin_xz: Tensor,
    frame_yaw: Tensor,
) -> Tensor:
    """Transform world-space XZ points into an origin-centered local frame."""
    _require_xz("points_world", points_world)
    origin = _expand_origin(origin_xz, points_world.ndim)
    return rotate_vectors_world_to_local(points_world - origin, frame_yaw)


def transform_points_local_to_world(
    points_local: Tensor,
    origin_xz: Tensor,
    frame_yaw: Tensor,
) -> Tensor:
    """Transform local XZ points into world coordinates."""
    _require_xz("points_local", points_local)
    origin = _expand_origin(origin_xz, points_local.ndim)
    return rotate_vectors_local_to_world(points_local, frame_yaw) + origin


__all__ = [
    "heading_to_direction",
    "matrix_to_yaw",
    "rotate_vectors_local_to_world",
    "rotate_vectors_world_to_local",
    "transform_points_local_to_world",
    "transform_points_world_to_local",
    "wrap_angle",
    "yaw_to_matrix",
]
