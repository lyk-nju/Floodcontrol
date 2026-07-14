"""Four-frame token/frame mapping.

Token ``k`` always covers the half-open frame range ``[4k, 4(k+1))``.
There is no first-token special case.
"""

from __future__ import annotations

FRAMES_PER_TOKEN_DEFAULT = 4


def _check_factor(frames_per_token: int) -> int:
    factor = int(frames_per_token)
    if factor <= 0:
        raise ValueError("frames_per_token must be positive")
    return factor


def token_start_frame(token_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    return max(0, int(token_idx)) * _check_factor(frames_per_token)


def token_end_frame(token_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    return token_start_frame(token_idx, frames_per_token) + _check_factor(frames_per_token) - 1


def commit_boundary_frame(commit_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    """Last available frame before commit_idx; cold start has no negative index."""
    return max(0, first_future_frame_abs(commit_idx, frames_per_token) - 1)


def first_future_frame_abs(commit_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    return max(0, int(commit_idx)) * _check_factor(frames_per_token)


def last_generated_frame_abs(commit_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int | None:
    future = first_future_frame_abs(commit_idx, frames_per_token)
    return None if future == 0 else future - 1


def num_frames_for_tokens(num_tokens: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    return max(0, int(num_tokens)) * _check_factor(frames_per_token)


def frame_idx_to_token_idx(frame_idx: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    return max(0, int(frame_idx)) // _check_factor(frames_per_token)


def num_tokens_for_frame_len(frame_len: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT) -> int:
    frames = max(0, int(frame_len))
    factor = _check_factor(frames_per_token)
    return (frames + factor - 1) // factor


def token_range_to_frame_slice(
    start_token_idx: int,
    num_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> slice:
    start = token_start_frame(start_token_idx, frames_per_token)
    return slice(start, start + num_frames_for_tokens(num_tokens, frames_per_token))


def token_active_window_left_frame(
    end_token_idx: int,
    chunk_size_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    return token_start_frame(max(0, int(end_token_idx) - int(chunk_size_tokens)), frames_per_token)


def token_body_window_left_frame(
    end_token_idx: int,
    body_window_tokens: int,
    frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT,
) -> int:
    return token_start_frame(max(0, int(end_token_idx) - int(body_window_tokens)), frames_per_token)


def frames_to_token_mask(mask_frame, num_tokens: int, frames_per_token: int = FRAMES_PER_TOKEN_DEFAULT):
    """OR-reduce each non-overlapping four-frame patch."""
    factor = _check_factor(frames_per_token)
    leading = mask_frame.shape[:-1]
    result = mask_frame.new_zeros(*leading, int(num_tokens))
    for token in range(int(num_tokens)):
        start = token * factor
        stop = min(start + factor, mask_frame.shape[-1])
        if start < stop:
            result[..., token] = (mask_frame[..., start:stop] > 0).any(-1).to(mask_frame.dtype)
    return result


def prefix_len_from_tail_invalid(token_mask):
    import torch

    valid = token_mask > 0
    _, num_tokens = valid.shape
    invalid = ~valid
    has_invalid = invalid.any(dim=1)
    first_invalid = torch.argmax(invalid.to(torch.int8), dim=1)
    prefix_len = torch.where(has_invalid, first_invalid, torch.full_like(first_invalid, num_tokens))
    pure_prefix = valid.sum(dim=1) == prefix_len
    return torch.where(pure_prefix, prefix_len, torch.full_like(prefix_len, num_tokens)).long()


__all__ = [
    "FRAMES_PER_TOKEN_DEFAULT", "commit_boundary_frame", "first_future_frame_abs",
    "frame_idx_to_token_idx", "frames_to_token_mask", "last_generated_frame_abs",
    "num_frames_for_tokens", "num_tokens_for_frame_len", "prefix_len_from_tail_invalid",
    "token_active_window_left_frame", "token_body_window_left_frame", "token_end_frame",
    "token_range_to_frame_slice", "token_start_frame",
]
