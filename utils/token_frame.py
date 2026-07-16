"""Canonical mapping between motion frames and latent tokens.

Floodcontrol has one temporal contract: every token represents exactly four
consecutive frames. Token ``k`` owns the half-open frame interval
``[4 * k, 4 * (k + 1))``. There is no special one-frame first token.

This module deliberately separates exact model-boundary conversions from
explicit preprocessing truncation. Model, Dataset and runtime code must use
the exact conversions; only offline preprocessing should call
``aligned_frame_floor()`` before discarding an incomplete tail patch.
"""

from __future__ import annotations

from numbers import Integral

import torch


FRAMES_PER_TOKEN = 4
MOTION_FPS = 20.0


def _non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def require_aligned_frame_count(frame_count: int) -> int:
    """Validate and return a frame count containing only complete patches."""
    frame_count = _non_negative_integer("frame_count", frame_count)
    if frame_count % FRAMES_PER_TOKEN:
        raise ValueError(
            f"frame_count must be divisible by four ({FRAMES_PER_TOKEN}), got {frame_count}"
        )
    return frame_count


def aligned_frame_floor(frame_count: int) -> int:
    """Return the largest complete-patch length not exceeding ``frame_count``.

    This operation may discard tail frames and is therefore intended only for
    explicit offline preprocessing code.
    """
    frame_count = _non_negative_integer("frame_count", frame_count)
    return frame_count - frame_count % FRAMES_PER_TOKEN


def frame_count_to_token_count(frame_count: int) -> int:
    """Convert an exactly aligned frame count to its token count."""
    return require_aligned_frame_count(frame_count) // FRAMES_PER_TOKEN


def token_count_to_frame_count(token_count: int) -> int:
    """Convert a token count to the exact number of represented frames."""
    return _non_negative_integer("token_count", token_count) * FRAMES_PER_TOKEN


def token_index_to_frame_start(token_index: int) -> int:
    """Return the first frame owned by ``token_index``."""
    return _non_negative_integer("token_index", token_index) * FRAMES_PER_TOKEN


def token_index_to_frame_slice(token_index: int) -> slice:
    """Return the four-frame half-open slice owned by one token."""
    start = token_index_to_frame_start(token_index)
    return slice(start, start + FRAMES_PER_TOKEN)


def token_range_to_frame_slice(start_token_index: int, token_count: int) -> slice:
    """Return the frame slice represented by a contiguous token range."""
    start = token_index_to_frame_start(start_token_index)
    return slice(start, start + token_count_to_frame_count(token_count))


def frame_index_to_token_index(frame_index: int) -> int:
    """Return the token that owns an absolute frame index."""
    return _non_negative_integer("frame_index", frame_index) // FRAMES_PER_TOKEN


def commit_index_to_frame_count(commit_index: int) -> int:
    """Return the number of frames committed before a token commit boundary."""
    return token_count_to_frame_count(commit_index)


def previous_committed_frame_index(commit_index: int) -> int | None:
    """Return the last committed frame, or ``None`` at cold start."""
    committed_frames = commit_index_to_frame_count(commit_index)
    return None if committed_frames == 0 else committed_frames - 1


def _patch_frame_mask(mask: torch.Tensor, *, frame_dim: int, reduce: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}")
    if mask.dtype != torch.bool:
        raise TypeError(f"mask must have dtype bool, got {mask.dtype}")
    if mask.ndim == 0:
        raise ValueError("mask must have at least one dimension")

    frame_dim = frame_dim % mask.ndim
    moved = mask.movedim(frame_dim, -1)
    frames = require_aligned_frame_count(moved.shape[-1])
    patched = moved.reshape(*moved.shape[:-1], frame_count_to_token_count(frames), FRAMES_PER_TOKEN)
    if reduce == "all":
        reduced = patched.all(dim=-1)
    elif reduce == "any":
        reduced = patched.any(dim=-1)
    else:  # pragma: no cover - private invariant
        raise ValueError(f"unknown reduction: {reduce}")
    return reduced.movedim(-1, frame_dim)


def frame_valid_to_token_valid(mask: torch.Tensor, *, frame_dim: int = -1) -> torch.Tensor:
    """AND-reduce frame validity so a token is valid only if all four frames are."""
    return _patch_frame_mask(mask, frame_dim=frame_dim, reduce="all")


def frame_observation_to_token_mask(
    mask: torch.Tensor,
    *,
    frame_dim: int = -1,
) -> torch.Tensor:
    """OR-reduce sparse observations so any observed frame activates the token."""
    return _patch_frame_mask(mask, frame_dim=frame_dim, reduce="any")


def prefix_valid_token_count(token_mask: torch.Tensor) -> torch.Tensor:
    """Count valid prefix tokens and reject masks containing validity holes.

    Args:
        token_mask: Boolean tensor whose final dimension is token time.

    Returns:
        Long tensor with the final token dimension removed.
    """
    if not isinstance(token_mask, torch.Tensor):
        raise TypeError("token_mask must be a torch.Tensor")
    if token_mask.dtype != torch.bool:
        raise TypeError(f"token_mask must have dtype bool, got {token_mask.dtype}")
    if token_mask.ndim == 0:
        raise ValueError("token_mask must have at least one dimension")
    if token_mask.shape[-1] > 1:
        has_hole = ((~token_mask[..., :-1]) & token_mask[..., 1:]).any()
        if bool(has_hole):
            raise ValueError("token_mask must be a contiguous valid prefix")
    return token_mask.sum(dim=-1, dtype=torch.long)


__all__ = [
    "FRAMES_PER_TOKEN",
    "MOTION_FPS",
    "aligned_frame_floor",
    "commit_index_to_frame_count",
    "frame_count_to_token_count",
    "frame_index_to_token_index",
    "frame_observation_to_token_mask",
    "frame_valid_to_token_valid",
    "prefix_valid_token_count",
    "previous_committed_frame_index",
    "require_aligned_frame_count",
    "token_count_to_frame_count",
    "token_index_to_frame_slice",
    "token_index_to_frame_start",
    "token_range_to_frame_slice",
]
