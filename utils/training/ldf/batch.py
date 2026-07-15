"""Model-facing batch construction for fixed-span LDF training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from utils.conditions.ldf import (
    HybridMotion,
    LDFCondition,
    LDFInput,
)
from utils.motion_process import ROOT_DIM
from utils.training.ldf.flow import (
    build_span_beta,
    flow_velocity_target,
    mix_fixed_noise,
)


@dataclass(frozen=True)
class LDFStepView:
    """Region and position metadata supplied to a condition builder."""

    step_index: int
    history_end: int
    active_start: int
    active_end: int
    frontier_start: int
    timeline_position_ids: torch.Tensor
    rope_position_ids: torch.Tensor
    beta: torch.Tensor


@dataclass(frozen=True)
class LDFTrainingStep:
    inputs: LDFInput
    target_velocity: HybridMotion
    loss_mask: torch.Tensor
    noise: HybridMotion
    view: LDFStepView


ConditionBuilder = Callable[[LDFStepView], LDFCondition]


def anchor_physical_batch(
    batch: dict[str, object],
    translation_anchor_xz: torch.Tensor,
) -> dict[str, object]:
    """Translate a physical span and its previous-root boundary once."""

    root = batch["root_motion"]
    if not torch.is_tensor(root) or root.ndim != 3 or root.shape[-1] != ROOT_DIM:
        raise ValueError("root_motion must be physical [B,F,5]")
    anchor = torch.as_tensor(
        translation_anchor_xz,
        device=root.device,
        dtype=root.dtype,
    )
    if tuple(anchor.shape) != (root.shape[0], 2):
        raise ValueError("translation_anchor_xz must be [B,2]")

    anchored = dict(batch)
    anchored_root = root.clone()
    anchored_root[..., 0] -= anchor[:, None, 0]
    anchored_root[..., 2] -= anchor[:, None, 1]
    anchored["root_motion"] = anchored_root

    previous = batch.get("previous_root_frame")
    previous_valid = batch.get("previous_root_valid_mask")
    if previous is not None:
        previous = previous.clone()
        if previous_valid is None:
            valid = torch.ones(root.shape[0], device=root.device, dtype=torch.bool)
        else:
            valid = previous_valid.to(device=root.device, dtype=torch.bool)
        previous[valid, 0] -= anchor[valid, 0]
        previous[valid, 2] -= anchor[valid, 1]
        anchored["previous_root_frame"] = previous
    return anchored


def build_ldf_training_step(
    *,
    clean_motion: HybridMotion,
    noise: HybridMotion,
    source_start_token: torch.Tensor,
    initial_history_tokens: int,
    active_tokens: int,
    phase_offset: torch.Tensor,
    step_index: int,
    previous_root_frame: torch.Tensor | None,
    previous_root_valid_mask: torch.Tensor | None,
    condition_builder: ConditionBuilder,
) -> LDFTrainingStep:
    """Build one fixed-S history/active/frontier LDF forward contract."""

    clean_motion.validate()
    batch = clean_motion.batch_size
    span_tokens = clean_motion.token_length
    source_start = source_start_token.to(
        device=clean_motion.root_motion.device,
        dtype=torch.long,
    ).view(-1)
    if tuple(source_start.shape) != (batch,):
        raise ValueError("source_start_token must be [B]")
    phase = phase_offset.to(
        device=clean_motion.root_motion.device,
        dtype=clean_motion.root_motion.dtype,
    )
    beta = build_span_beta(
        span_tokens=span_tokens,
        initial_history_tokens=initial_history_tokens,
        active_tokens=active_tokens,
        phase_offset=phase,
        step_index=step_index,
    )
    history_end = int(initial_history_tokens) + int(step_index)
    active_start = history_end
    active_end = active_start + int(active_tokens)
    positions = torch.arange(
        span_tokens,
        device=clean_motion.root_motion.device,
        dtype=torch.long,
    )[None].expand(batch, -1)
    timeline = source_start[:, None] + positions
    rope = positions - active_start
    history_mask = positions < active_start
    loss_mask = (positions >= active_start) & (positions < active_end)
    # Pure-noise frontier remains in persistent state for later rollout steps,
    # but only history plus the current active band enters non-causal attention.
    generation_mask = loss_mask
    noisy_motion = mix_fixed_noise(clean_motion, noise, beta)
    view = LDFStepView(
        step_index=int(step_index),
        history_end=history_end,
        active_start=active_start,
        active_end=active_end,
        frontier_start=active_end,
        timeline_position_ids=timeline,
        rope_position_ids=rope,
        beta=beta,
    )
    condition = condition_builder(view)
    if not isinstance(condition, LDFCondition):
        raise TypeError("condition_builder must return LDFCondition")
    inputs = LDFInput(
        noisy_motion=noisy_motion,
        beta=beta,
        history_mask=history_mask,
        generation_mask=generation_mask,
        timeline_position_ids=timeline,
        rope_position_ids=rope,
        previous_root_frame=previous_root_frame,
        previous_root_valid_mask=previous_root_valid_mask,
        condition=condition,
    )
    inputs.validate()
    return LDFTrainingStep(
        inputs=inputs,
        target_velocity=flow_velocity_target(clean_motion, noise),
        loss_mask=loss_mask,
        noise=noise,
        view=view,
    )


__all__ = [
    "ConditionBuilder",
    "LDFStepView",
    "LDFTrainingStep",
    "anchor_physical_batch",
    "build_ldf_training_step",
]
