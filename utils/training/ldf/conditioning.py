"""Training-side sampling and compilation of explicit XZ constraints."""

from __future__ import annotations

import torch

from utils.conditions.ldf import LDFCondition, create_ldf_condition
from utils.motion_process import ROOT_DIM
from utils.token_frame import FRAMES_PER_TOKEN
from utils.training.ldf.steps import LDFStepView


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


def sample_future_horizon_tokens(
    *,
    token_valid_mask: torch.Tensor,
    initial_active_end: torch.Tensor,
    rollout_steps: int,
    max_horizon_token: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one persistent per-sample future lookahead in ``[0, max]``.

    The sampled horizon must remain fully available after the final rollout
    commit, so shorter samples and larger K values reduce the per-sample upper
    bound.  A zero horizon means that no not-yet-visible XZ token is exposed;
    it is distinct from constraint dropout, which also removes visible XZ.
    """

    if (
        not torch.is_tensor(token_valid_mask)
        or token_valid_mask.ndim != 2
        or token_valid_mask.dtype != torch.bool
    ):
        raise ValueError("token_valid_mask must be bool [B,T]")
    batch = token_valid_mask.shape[0]
    active_end = torch.as_tensor(
        initial_active_end,
        device=token_valid_mask.device,
        dtype=torch.long,
    ).reshape(-1)
    rollout_steps = int(rollout_steps)
    maximum = int(max_horizon_token)
    if tuple(active_end.shape) != (batch,):
        raise ValueError("initial_active_end must be [B]")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if maximum < 0:
        raise ValueError("max_horizon_token must be non-negative")

    valid_counts = token_valid_mask.sum(dim=1, dtype=torch.long)
    available = valid_counts - active_end - (rollout_steps - 1)
    if bool((available < 0).any()):
        raise ValueError("active window leaves insufficient rollout frontier")
    upper = torch.minimum(available, torch.full_like(available, maximum))
    draws = torch.rand(
        batch,
        device=token_valid_mask.device,
        generator=generator,
    )
    return torch.floor(draws * (upper + 1).to(draws.dtype)).to(torch.long)


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
    future_horizon_tokens: torch.Tensor,
    rollout_steps: int,
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
    horizon = torch.as_tensor(
        future_horizon_tokens,
        device=token_valid_mask.device,
        dtype=torch.long,
    ).reshape(-1)
    rollout_steps = int(rollout_steps)
    max_waypoint_count = int(max_waypoint_count)
    if (
        tuple(active_start.shape) != (batch,)
        or tuple(active_end.shape) != (batch,)
        or tuple(horizon.shape) != (batch,)
    ):
        raise ValueError("initial active bounds and future horizon must be [B]")
    valid_counts = token_valid_mask.sum(dim=1, dtype=torch.long)
    if bool((horizon < 0).any()):
        raise ValueError("future_horizon_tokens must be non-negative")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
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
    bounds = torch.stack(
        (active_start, active_end, valid_counts, horizon), dim=1
    ).tolist()
    mode_draws_list = mode_draws.tolist()
    rows = zip(bounds, mode_draws_list, strict=True)
    for batch_index, (
        (sample_active_start, sample_active_end, valid_count, sample_horizon),
        draw,
    ) in enumerate(rows):
        # The immutable plan must cover the lookahead after every later commit,
        # not only after the initial active band.  Per-step compilation still
        # exposes at most ``sample_horizon`` tokens from the current boundary.
        future_end = min(
            valid_count,
            sample_active_end + sample_horizon + rollout_steps - 1,
        )
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


def create_xz_condition(
    *,
    clean_root_motion: torch.Tensor,
    token_valid_mask: torch.Tensor,
    constraint_mask: torch.Tensor,
    view: LDFStepView,
    text_context: list[torch.Tensor],
    text_null_context: list[torch.Tensor],
    future_horizon_tokens: torch.Tensor,
) -> LDFCondition:
    """Compile one persistent absolute XZ plan for the current rollout view."""

    if (
        not torch.is_tensor(clean_root_motion)
        or clean_root_motion.ndim != 4
        or tuple(clean_root_motion.shape[2:]) != (FRAMES_PER_TOKEN, ROOT_DIM)
    ):
        raise ValueError("clean_root_motion must be physical [B,T,4,5]")
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
    horizon = torch.as_tensor(
        future_horizon_tokens,
        device=clean_root_motion.device,
        dtype=torch.long,
    ).reshape(-1)
    if tuple(horizon.shape) != (batch,):
        raise ValueError("future_horizon_tokens must be [B]")
    if bool((horizon < 0).any()):
        raise ValueError("future_horizon_tokens must be non-negative")
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

    # Compile one immutable candidate superset for the whole commit.  The Root
    # Stage will move the effective boundary from the first currently-visible
    # motion token through the active band as triangular denoising progresses.
    future_start = view.active_start + 1
    requested_future_end = torch.where(
        horizon > 0,
        view.active_end + horizon,
        future_start,
    )
    future_end = torch.minimum(
        valid_counts,
        requested_future_end,
    )
    candidate_range = (
        (positions >= future_start[:, None])
        & (positions < future_end[:, None])
    )
    future_mask = mask & candidate_range[..., None, None]

    return create_ldf_condition(
        text_context=text_context,
        text_null_context=text_null_context,
        root_condition_value=clean_root_motion.detach(),
        root_condition_mask=active_mask,
        future_root_condition_value=clean_root_motion.detach(),
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=view.timeline_position_ids,
        future_horizon_tokens=horizon,
    )


__all__ = [
    "create_xz_condition",
    "sample_constraint_keep_mask",
    "sample_future_horizon_tokens",
    "sample_xz_constraint_mask",
]
