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
    initial_active_start: int,
    initial_active_end: int,
    future_lookahead_tokens: int,
    dense_probability: float,
    waypoint_probability: float,
    goal_probability: float,
    max_waypoints: int,
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
    active_start = int(initial_active_start)
    active_end = int(initial_active_end)
    lookahead = int(future_lookahead_tokens)
    max_waypoints = int(max_waypoints)
    if not 0 <= active_start < active_end <= tokens:
        raise ValueError("initial active range lies outside token_valid_mask")
    if lookahead < 0:
        raise ValueError("future_lookahead_tokens must be non-negative")
    if max_waypoints <= 0:
        raise ValueError("max_waypoints must be positive")
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
    future_end = min(tokens, active_end + lookahead)

    for batch_index in range(batch):
        valid_tokens = torch.nonzero(
            token_valid_mask[batch_index], as_tuple=False
        ).flatten()
        eligible_tokens = valid_tokens[
            (valid_tokens >= active_start) & (valid_tokens < future_end)
        ]
        if eligible_tokens.numel() == 0:
            raise ValueError("every sample must contain a valid active constraint frame")
        eligible_frames = (
            eligible_tokens[:, None] * FRAMES_PER_TOKEN
            + torch.arange(
                FRAMES_PER_TOKEN, device=eligible_tokens.device
            )[None]
        ).flatten()

        draw = float(mode_draws[batch_index].item())
        if draw < dense:
            dense_tokens = valid_tokens[valid_tokens >= active_start]
            dense_frames = (
                dense_tokens[:, None] * FRAMES_PER_TOKEN
                + torch.arange(
                    FRAMES_PER_TOKEN, device=dense_tokens.device
                )[None]
            ).flatten()
            _mark_xz(plan, batch_index, dense_frames)
            continue

        if draw < dense + waypoint:
            count_limit = min(max_waypoints, int(eligible_frames.numel()))
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

        future_tokens = valid_tokens[
            (valid_tokens >= active_end) & (valid_tokens < future_end)
        ]
        if future_tokens.numel() > 0:
            goal_token_offset = int(
                torch.randint(
                    future_tokens.numel(),
                    (),
                    device=future_tokens.device,
                    generator=generator,
                ).item()
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
                eligible_frames < active_end * FRAMES_PER_TOKEN
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

    if bool((plan[..., 0] != plan[..., 2]).any()):
        raise RuntimeError("XZ constraint features must always be observed together")
    if bool(plan[..., 1].any()) or bool(plan[..., 3:].any()):
        raise RuntimeError("trajectory sampling may only expose XZ")
    return plan


def _pack_future_constraints(
    *,
    clean_root_motion: torch.Tensor,
    constraint_mask: torch.Tensor,
    timeline_position_ids: torch.Tensor,
    future_start: int,
    future_end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    candidate_mask = constraint_mask[:, future_start:future_end]
    candidate_valid = candidate_mask.flatten(2).any(dim=-1)
    counts = candidate_valid.sum(dim=1, dtype=torch.long)
    packed_tokens = int(counts.max().item())
    if packed_tokens == 0:
        return None

    batch = clean_root_motion.shape[0]
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
    for batch_index in range(batch):
        selected = torch.nonzero(
            candidate_valid[batch_index], as_tuple=False
        ).flatten()
        count = int(selected.numel())
        if count == 0:
            continue
        source_indices = selected + future_start
        future_value[batch_index, :count] = clean_root_motion[
            batch_index, source_indices
        ]
        future_mask[batch_index, :count] = constraint_mask[
            batch_index, source_indices
        ]
        future_positions[batch_index, :count] = timeline_position_ids[
            batch_index, source_indices
        ]
        future_valid[batch_index, :count] = True
    return future_value, future_mask, future_positions, future_valid


def create_xz_condition(
    *,
    clean_root_motion: torch.Tensor,
    token_valid_mask: torch.Tensor,
    constraint_mask: torch.Tensor,
    view: LDFStepView,
    text_context: list[torch.Tensor],
    text_null_context: list[torch.Tensor],
    future_lookahead_tokens: int,
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
    if bool((mask[..., 0] != mask[..., 2]).any()):
        raise ValueError("constraint_mask must observe XZ together")
    if bool(mask[..., 1].any()) or bool(mask[..., 3:].any()):
        raise ValueError("constraint_mask may only observe XZ")
    mask = mask & token_valid_mask[..., None, None]
    lookahead = int(future_lookahead_tokens)
    if lookahead < 0:
        raise ValueError("future_lookahead_tokens must be non-negative")
    if not 0 <= view.active_start < view.active_end <= tokens:
        raise ValueError("active range lies outside clean_root_motion")
    if tuple(view.timeline_position_ids.shape) != (batch, tokens):
        raise ValueError("timeline_position_ids must match clean root [B,T]")

    active_mask = torch.zeros_like(mask)
    active_mask[:, view.active_start : view.active_end] = mask[
        :, view.active_start : view.active_end
    ]

    future_start = view.active_end
    future_end = min(tokens, future_start + lookahead)
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
    condition.validate(batch_size=batch, token_length=tokens)
    return condition


__all__ = [
    "create_xz_condition",
    "sample_constraint_keep_mask",
    "sample_xz_constraint_mask",
]
