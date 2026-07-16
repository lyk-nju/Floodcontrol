"""Training-side sampling and compilation of explicit XZ constraints."""

from __future__ import annotations

import torch

from utils.conditions.ldf import LDFCondition
from utils.motion_process import ROOT_DIM
from utils.token_frame import FRAMES_PER_TOKEN
from utils.training.ldf.batch import LDFStepView


_XZ_FEATURES = (0, 2)


def sample_constraint_keep_mask(
    batch_size: int,
    *,
    dropout_probability: float,
    device: torch.device,
    generator: torch.Generator | None = None,
    apply_dropout: bool,
) -> torch.Tensor:
    """Sample one independent constraint-CFG decision per batch item."""

    batch_size = int(batch_size)
    probability = float(dropout_probability)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("constraint dropout probability must lie in [0,1]")
    if not apply_dropout or probability == 0.0:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    if probability == 1.0:
        return torch.zeros(batch_size, device=device, dtype=torch.bool)
    return torch.rand(batch_size, device=device, generator=generator) >= probability


def _validate_sampling_probabilities(
    dense_probability: float,
    waypoint_probability: float,
    goal_probability: float,
) -> tuple[float, float, float]:
    probabilities = tuple(
        float(value)
        for value in (
            dense_probability,
            waypoint_probability,
            goal_probability,
        )
    )
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("XZ constraint sampling probabilities must lie in [0,1]")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        raise ValueError("XZ constraint sampling probabilities must sum to one")
    return probabilities


def _mark_xz(mask: torch.Tensor, batch_index: int, flat_frame_indices: torch.Tensor):
    token_indices = torch.div(
        flat_frame_indices, FRAMES_PER_TOKEN, rounding_mode="floor"
    )
    frame_indices = flat_frame_indices.remainder(FRAMES_PER_TOKEN)
    for feature in _XZ_FEATURES:
        mask[batch_index, token_indices, frame_indices, feature] = True


def sample_xz_constraint_mask(
    *,
    token_valid_mask: torch.Tensor,
    initial_active_start: torch.Tensor,
    initial_active_end: torch.Tensor,
    max_horizon_token: int,
    dense_probability: float,
    waypoint_probability: float,
    goal_probability: float,
    max_waypoint_count: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one persistent absolute XZ plan for an entire rollout.

    Dense samples expose every valid frame from the first active token onward.
    Waypoint samples expose a small number of individual frames across the first
    active band and its future lookahead. Goal samples expose exactly one frame
    strictly after the first active band. If a short sample has no future frame,
    goal sampling falls back to one active waypoint; constraint dropout remains
    the only mechanism that creates an entirely unconstrained sample.
    """

    if (
        not torch.is_tensor(token_valid_mask)
        or token_valid_mask.ndim != 2
        or token_valid_mask.dtype != torch.bool
    ):
        raise ValueError("token_valid_mask must be bool [B,T]")
    batch, tokens = token_valid_mask.shape
    active_start = torch.as_tensor(
        initial_active_start,
        device=token_valid_mask.device,
        dtype=torch.long,
    ).reshape(-1)
    active_end = torch.as_tensor(
        initial_active_end,
        device=token_valid_mask.device,
        dtype=torch.long,
    ).reshape(-1)
    lookahead = int(max_horizon_token)
    max_waypoint_count = int(max_waypoint_count)
    if tuple(active_start.shape) != (batch,) or tuple(active_end.shape) != (batch,):
        raise ValueError("initial active bounds must be [B]")
    valid_counts = token_valid_mask.sum(dim=1, dtype=torch.long)
    if lookahead < 0:
        raise ValueError("max_horizon_token must be non-negative")
    if max_waypoint_count <= 0:
        raise ValueError("max_waypoint_count must be positive")
    dense, waypoint, _goal = _validate_sampling_probabilities(
        dense_probability,
        waypoint_probability,
        goal_probability,
    )

    plan = torch.zeros(
        batch,
        tokens,
        FRAMES_PER_TOKEN,
        ROOT_DIM,
        device=token_valid_mask.device,
        dtype=torch.bool,
    )
    mode_draws = torch.rand(
        batch, device=token_valid_mask.device, generator=generator
    )
    # Variable-size waypoint/goal selection still uses a Python loop, but copy
    # its scalar control data to CPU once rather than synchronizing repeatedly
    # for every sample.
    bounds = torch.stack((active_start, active_end, valid_counts), dim=1).tolist()
    mode_draws_list = mode_draws.tolist()
    rows = zip(bounds, mode_draws_list, strict=True)
    for batch_index, (
        (sample_active_start, sample_active_end, valid_count),
        draw,
    ) in enumerate(rows):
        future_end = min(valid_count, sample_active_end + lookahead)
        eligible_tokens = torch.arange(
            sample_active_start,
            future_end,
            device=token_valid_mask.device,
        )
        eligible_frames = (
            eligible_tokens[:, None] * FRAMES_PER_TOKEN
            + torch.arange(
                FRAMES_PER_TOKEN, device=eligible_tokens.device
            )[None]
        ).flatten()

        if draw < dense:
            dense_frames = (
                eligible_tokens[:, None] * FRAMES_PER_TOKEN
                + torch.arange(
                    FRAMES_PER_TOKEN, device=eligible_tokens.device
                )[None]
            ).flatten()
            _mark_xz(plan, batch_index, dense_frames)
            continue

        if draw < dense + waypoint:
            count_limit = min(max_waypoint_count, int(eligible_frames.numel()))
            count = int(
                torch.randint(
                    1,
                    count_limit + 1,
                    (),
                    device=eligible_frames.device,
                    generator=generator,
                ).item()
            )
            order = torch.randperm(
                eligible_frames.numel(),
                device=eligible_frames.device,
                generator=generator,
            )[:count]
            _mark_xz(plan, batch_index, eligible_frames[order])
            continue

        future_tokens = torch.arange(
            sample_active_end,
            future_end,
            device=token_valid_mask.device,
        )
        if future_tokens.numel() > 0:
            goal_token_offset = torch.randint(
                future_tokens.numel(),
                (),
                device=future_tokens.device,
                generator=generator,
            )
            goal_token = future_tokens[goal_token_offset]
            goal_frame = torch.randint(
                FRAMES_PER_TOKEN,
                (),
                device=future_tokens.device,
                generator=generator,
            )
            selected = goal_token * FRAMES_PER_TOKEN + goal_frame
        else:
            active_frames = eligible_frames[
                eligible_frames < sample_active_end * FRAMES_PER_TOKEN
            ]
            selected = active_frames[
                torch.randint(
                    active_frames.numel(),
                    (),
                    device=active_frames.device,
                    generator=generator,
                )
            ]
        _mark_xz(plan, batch_index, selected.reshape(1))

    return plan


def _pack_future_constraints(
    *,
    clean_root_motion: torch.Tensor,
    constraint_mask: torch.Tensor,
    timeline_position_ids: torch.Tensor,
    future_start: torch.Tensor,
    future_end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    batch, tokens = constraint_mask.shape[:2]
    positions = torch.arange(tokens, device=constraint_mask.device)[None]
    candidate_range = (
        (positions >= future_start[:, None])
        & (positions < future_end[:, None])
    )
    candidate_valid = constraint_mask.flatten(2).any(dim=-1) & candidate_range
    counts = candidate_valid.sum(dim=1, dtype=torch.long)
    packed_tokens = int(counts.max().item())
    if packed_tokens == 0:
        return None

    future_value = clean_root_motion.new_zeros(
        batch, packed_tokens, FRAMES_PER_TOKEN, ROOT_DIM
    )
    future_mask = torch.zeros_like(future_value, dtype=torch.bool)
    future_positions = timeline_position_ids.new_zeros(batch, packed_tokens)
    future_valid = torch.zeros(
        batch,
        packed_tokens,
        device=clean_root_motion.device,
        dtype=torch.bool,
    )
    packed_indices = candidate_valid.cumsum(dim=1, dtype=torch.long) - 1
    batch_indices = torch.arange(
        batch, device=constraint_mask.device
    )[:, None].expand(-1, tokens)
    source_indices = positions.expand(batch, -1)
    selected_batch = batch_indices[candidate_valid]
    selected_source = source_indices[candidate_valid]
    selected_destination = packed_indices[candidate_valid]
    future_value[selected_batch, selected_destination] = clean_root_motion[
        selected_batch, selected_source
    ]
    future_mask[selected_batch, selected_destination] = constraint_mask[
        selected_batch, selected_source
    ]
    future_positions[selected_batch, selected_destination] = timeline_position_ids[
        selected_batch, selected_source
    ]
    future_valid[selected_batch, selected_destination] = True
    return future_value, future_mask, future_positions, future_valid


def create_xz_condition(
    *,
    clean_root_motion: torch.Tensor,
    token_valid_mask: torch.Tensor,
    constraint_mask: torch.Tensor,
    view: LDFStepView,
    text_context: list[torch.Tensor],
    text_null_context: list[torch.Tensor],
    max_horizon_token: int,
) -> LDFCondition:
    """Compile one persistent absolute XZ plan for the current rollout view."""

    if (
        not torch.is_tensor(clean_root_motion)
        or clean_root_motion.ndim != 4
        or tuple(clean_root_motion.shape[2:]) != (FRAMES_PER_TOKEN, ROOT_DIM)
    ):
        raise ValueError("clean_root_motion must be normalized [B,T,4,5]")
    batch, tokens = clean_root_motion.shape[:2]
    if (
        not torch.is_tensor(token_valid_mask)
        or tuple(token_valid_mask.shape) != (batch, tokens)
        or token_valid_mask.dtype != torch.bool
    ):
        raise ValueError("token_valid_mask must be bool [B,T]")
    mask = constraint_mask.to(device=clean_root_motion.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(clean_root_motion.shape):
        raise ValueError("constraint_mask must match clean_root_motion")
    mask = mask & token_valid_mask[..., None, None]
    lookahead = int(max_horizon_token)
    if lookahead < 0:
        raise ValueError("max_horizon_token must be non-negative")
    for name, value in (
        ("active_start", view.active_start),
        ("active_end", view.active_end),
    ):
        if not torch.is_tensor(value) or tuple(value.shape) != (batch,):
            raise ValueError(f"view.{name} must be [B]")
    valid_counts = token_valid_mask.sum(dim=1, dtype=torch.long)
    if tuple(view.timeline_position_ids.shape) != (batch, tokens):
        raise ValueError("timeline_position_ids must match clean root [B,T]")

    positions = torch.arange(tokens, device=mask.device)[None]
    active_range = (
        (positions >= view.active_start[:, None])
        & (positions < view.active_end[:, None])
    )
    active_mask = mask & active_range[..., None, None]

    future_start = view.active_end
    future_end = torch.minimum(
        valid_counts,
        future_start + lookahead,
    )
    packed = _pack_future_constraints(
        clean_root_motion=clean_root_motion,
        constraint_mask=mask,
        timeline_position_ids=view.timeline_position_ids,
        future_start=future_start,
        future_end=future_end,
    )
    future_value = future_mask = future_positions = future_valid = None
    if packed is not None:
        future_value, future_mask, future_positions, future_valid = packed

    condition = LDFCondition(
        text_context=list(text_context),
        text_null_context=list(text_null_context),
        root_condition_value=clean_root_motion.detach(),
        root_condition_mask=active_mask,
        future_root_condition_value=future_value,
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=future_positions,
        future_valid_mask=future_valid,
    )
    condition.validate_structure(batch_size=batch, token_length=tokens)
    return condition


__all__ = [
    "create_xz_condition",
    "sample_constraint_keep_mask",
    "sample_xz_constraint_mask",
]
