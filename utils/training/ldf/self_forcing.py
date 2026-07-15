"""Detached fixed-span self-forcing for LDF training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from utils.conditions.ldf import HybridMotion, LDFPrediction
from utils.token_frame import FRAMES_PER_TOKEN
from utils.training.ldf.batch import (
    ConditionBuilder,
    LDFTrainingStep,
    build_ldf_training_step,
)
from utils.training.ldf.flow import recover_clean_for_self_forcing


DEFAULT_K_SCHEDULE = ((0.0, 2), (0.4, 3), (0.7, 5))
DEFAULT_TEACHER_REPLAY = {2: 0.2, 3: 0.1, 5: 0.1}


@dataclass(frozen=True)
class LDFWindowPlan:
    """Immutable geometry, coordinate frame and noise for one rollout."""

    span_tokens: int
    initial_history_tokens: int
    active_tokens: int
    frontier_tokens: int
    rollout_steps: int
    source_start_token: torch.Tensor
    phase_offset: torch.Tensor
    translation_anchor_frame: torch.Tensor
    translation_anchor_xz: torch.Tensor
    root_noise: torch.Tensor
    body_noise: torch.Tensor
    cold_start_mask: torch.Tensor

    @property
    def noise(self) -> HybridMotion:
        return HybridMotion(self.root_noise, self.body_noise)

    def validate(self) -> None:
        batch = int(self.root_noise.shape[0])
        if self.span_tokens <= 0 or self.active_tokens <= 0:
            raise ValueError("span_tokens and active_tokens must be positive")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if self.initial_history_tokens < 0 or self.frontier_tokens < 0:
            raise ValueError("history/frontier lengths must be non-negative")
        if (
            self.initial_history_tokens
            + self.active_tokens
            + self.frontier_tokens
            != self.span_tokens
        ):
            raise ValueError("S must equal H + active + frontier")
        if self.frontier_tokens < self.rollout_steps - 1:
            raise ValueError("frontier cannot support the requested rollout depth")
        expected_root = (batch, self.span_tokens, FRAMES_PER_TOKEN, 5)
        if tuple(self.root_noise.shape) != expected_root:
            raise ValueError(f"root_noise must be {expected_root}")
        if self.body_noise.ndim != 3 or tuple(self.body_noise.shape[:2]) != (
            batch,
            self.span_tokens,
        ):
            raise ValueError("body_noise must be [B,S,D]")
        for name, value, shape in (
            ("source_start_token", self.source_start_token, (batch,)),
            ("phase_offset", self.phase_offset, (batch,)),
            ("translation_anchor_frame", self.translation_anchor_frame, (batch,)),
            ("translation_anchor_xz", self.translation_anchor_xz, (batch, 2)),
            ("cold_start_mask", self.cold_start_mask, (batch,)),
        ):
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.source_start_token.dtype != torch.long:
            raise TypeError("source_start_token must be long")
        if self.translation_anchor_frame.dtype != torch.long:
            raise TypeError("translation_anchor_frame must be long")
        if self.cold_start_mask.dtype != torch.bool:
            raise TypeError("cold_start_mask must be bool")
        if bool((self.cold_start_mask != self.cold_start_mask[:1]).any()):
            raise ValueError("cold-start mode must be batch-shared")
        if bool(self.cold_start_mask[0]):
            if self.initial_history_tokens != 0 or bool(
                (self.source_start_token != 0).any()
            ):
                raise ValueError("true cold start requires H=0 and source start zero")
        elif self.initial_history_tokens < 1:
            raise ValueError("continuation windows require at least one history token")
        self.noise.validate()


@dataclass
class SelfForcingState:
    """Only the clean history substitutions are mutable across rollout steps."""

    clean_motion: HybridMotion
    completed_steps: int = 0

    def validate(self, plan: LDFWindowPlan) -> None:
        self.clean_motion.validate()
        if self.clean_motion.token_length != plan.span_tokens:
            raise ValueError("clean motion does not match the rollout span")
        if self.clean_motion.batch_size != plan.root_noise.shape[0]:
            raise ValueError("clean motion does not match the rollout batch")
        if not 0 <= self.completed_steps < plan.rollout_steps:
            raise ValueError("completed_steps lies outside the rollout")

    def replace_committed_token(
        self,
        *,
        token_index: int,
        root_motion: torch.Tensor,
        latent_motion: torch.Tensor,
    ) -> "SelfForcingState":
        root = self.clean_motion.root_motion.clone()
        latent = self.clean_motion.latent_motion.clone()
        root[:, token_index] = root_motion.detach()
        latent[:, token_index] = latent_motion.detach()
        return SelfForcingState(
            HybridMotion(root.detach(), latent.detach()),
            completed_steps=self.completed_steps + 1,
        )


@dataclass(frozen=True)
class SelfForcingResult:
    final_step: LDFTrainingStep
    prediction: LDFPrediction
    state: SelfForcingState
    replacements: tuple[HybridMotion, ...]


def resolve_self_forcing_k(
    progress: float,
    schedule=DEFAULT_K_SCHEDULE,
) -> int:
    """Resolve the current K from monotonically increasing progress thresholds."""

    progress = float(progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0,1]")
    rows = sorted((float(threshold), int(k)) for threshold, k in schedule)
    if not rows or rows[0][0] != 0.0 or any(k < 2 for _, k in rows):
        raise ValueError("self-forcing schedule must start at 0 with K>=2")
    selected = rows[0][1]
    for threshold, candidate in rows:
        if progress < threshold:
            break
        selected = candidate
    return selected


def sample_rollout_steps(
    progress: float,
    *,
    generator: torch.Generator | None = None,
    schedule=DEFAULT_K_SCHEDULE,
    teacher_replay: Mapping[int, float] | None = DEFAULT_TEACHER_REPLAY,
) -> int:
    """Sample K with configurable teacher-forcing replay during fine-tuning."""

    rollout_steps = resolve_self_forcing_k(progress, schedule)
    replay_probability = 0.0 if teacher_replay is None else float(
        teacher_replay.get(rollout_steps, 0.0)
    )
    if not 0.0 <= replay_probability <= 1.0:
        raise ValueError("teacher replay probability must lie in [0,1]")
    draw = float(torch.rand((), generator=generator).item())
    return 1 if draw < replay_probability else rollout_steps


def sample_window_plan(
    batch: dict[str, object],
    *,
    active_tokens: int,
    rollout_steps: int,
    latent_dim: int,
    generator: torch.Generator | None = None,
    initial_history_tokens: int | None = None,
    phase_offset: torch.Tensor | None = None,
    root_noise: torch.Tensor | None = None,
    body_noise: torch.Tensor | None = None,
) -> LDFWindowPlan:
    """Sample H, phase and fixed absolute-token noise for one source span."""

    root = batch["root_motion"]
    if not torch.is_tensor(root) or root.ndim != 3 or root.shape[-1] != 5:
        raise ValueError("root_motion must be physical [B,F,5]")
    batch_size, frames = root.shape[:2]
    if frames % FRAMES_PER_TOKEN:
        raise ValueError("source span must be four-frame aligned")
    span_tokens = frames // FRAMES_PER_TOKEN
    active_tokens = int(active_tokens)
    rollout_steps = int(rollout_steps)
    latent_dim = int(latent_dim)
    if active_tokens <= 0 or rollout_steps <= 0 or latent_dim <= 0:
        raise ValueError("active_tokens, rollout_steps and latent_dim must be positive")

    source_start = batch["source_start_token"].to(device=root.device, dtype=torch.long)
    cold = batch["cold_start_mask"].to(device=root.device, dtype=torch.bool)
    if tuple(source_start.shape) != (batch_size,) or tuple(cold.shape) != (batch_size,):
        raise ValueError("source_start_token and cold_start_mask must be [B]")
    if bool((cold != cold[:1]).any()):
        raise ValueError("cold-start mode must be batch-shared")

    maximum_history = span_tokens - active_tokens - (rollout_steps - 1)
    minimum_history = 0 if bool(cold[0]) else 1
    if maximum_history < minimum_history:
        raise ValueError("source span is too short for active band and rollout")
    if initial_history_tokens is None:
        if bool(cold[0]):
            history = 0
        elif minimum_history == maximum_history:
            history = minimum_history
        else:
            history = int(
                torch.randint(
                    minimum_history,
                    maximum_history + 1,
                    (),
                    device=root.device,
                    generator=generator,
                ).item()
            )
    else:
        history = int(initial_history_tokens)
    if bool(cold[0]) and history != 0:
        raise ValueError("true cold start requires H=0")
    if not bool(cold[0]) and history < 1:
        raise ValueError("continuation requires H>=1")
    if history > maximum_history:
        raise ValueError("history leaves insufficient rollout frontier")
    frontier = span_tokens - history - active_tokens

    anchor_frame_value = history * FRAMES_PER_TOKEN - 1 if history else 0
    anchor_frame = torch.full(
        (batch_size,), anchor_frame_value, device=root.device, dtype=torch.long
    )
    batch_indices = torch.arange(batch_size, device=root.device)
    anchor_xz = root[batch_indices, anchor_frame][:, [0, 2]].clone()

    if phase_offset is None:
        phase = torch.rand(
            batch_size,
            device=root.device,
            dtype=root.dtype,
            generator=generator,
        ) / float(active_tokens)
    else:
        phase = phase_offset.to(device=root.device, dtype=root.dtype).clone()
    if root_noise is None:
        root_noise = torch.randn(
            batch_size,
            span_tokens,
            FRAMES_PER_TOKEN,
            5,
            device=root.device,
            dtype=root.dtype,
            generator=generator,
        )
    if body_noise is None:
        body_noise = torch.randn(
            batch_size,
            span_tokens,
            latent_dim,
            device=root.device,
            dtype=root.dtype,
            generator=generator,
        )

    plan = LDFWindowPlan(
        span_tokens=span_tokens,
        initial_history_tokens=history,
        active_tokens=active_tokens,
        frontier_tokens=frontier,
        rollout_steps=rollout_steps,
        source_start_token=source_start.clone(),
        phase_offset=phase,
        translation_anchor_frame=anchor_frame,
        translation_anchor_xz=anchor_xz,
        root_noise=root_noise,
        body_noise=body_noise,
        cold_start_mask=cold.clone(),
    )
    plan.validate()
    return plan


def run_self_forcing_rollout(
    model: nn.Module,
    state: SelfForcingState,
    plan: LDFWindowPlan,
    *,
    previous_root_frame: torch.Tensor | None,
    previous_root_valid_mask: torch.Tensor | None,
    condition_builder: ConditionBuilder,
) -> SelfForcingResult:
    """Run K-1 detached replacements and one differentiable final forward."""

    plan.validate()
    state.validate(plan)
    replacements: list[HybridMotion] = []
    final_step = None
    final_prediction = None
    current = state

    for step_index in range(plan.rollout_steps):
        training_step = build_ldf_training_step(
            clean_motion=current.clean_motion,
            noise=plan.noise,
            source_start_token=plan.source_start_token,
            initial_history_tokens=plan.initial_history_tokens,
            active_tokens=plan.active_tokens,
            phase_offset=plan.phase_offset,
            step_index=step_index,
            previous_root_frame=previous_root_frame,
            previous_root_valid_mask=previous_root_valid_mask,
            condition_builder=condition_builder,
        )
        if step_index == plan.rollout_steps - 1:
            final_step = training_step
            final_prediction = model(training_step.inputs)
            break

        with torch.no_grad():
            prediction = model(training_step.inputs)
        replace_index = plan.initial_history_tokens + step_index
        root_replacement = prediction.clean_root_motion[:, replace_index].detach()
        body_replacement = recover_clean_for_self_forcing(
            training_step.inputs.noisy_motion.latent_motion[:, replace_index],
            training_step.inputs.beta[:, replace_index],
            prediction.velocity.latent_motion[:, replace_index],
        ).detach()
        replacement = HybridMotion(
            root_replacement[:, None],
            body_replacement[:, None],
        )
        replacement.validate()
        replacements.append(replacement)
        current = current.replace_committed_token(
            token_index=replace_index,
            root_motion=root_replacement,
            latent_motion=body_replacement,
        )

    if final_step is None or final_prediction is None:
        raise RuntimeError("self-forcing rollout did not produce a final step")
    return SelfForcingResult(
        final_step=final_step,
        prediction=final_prediction,
        state=current,
        replacements=tuple(replacements),
    )


__all__ = [
    "DEFAULT_K_SCHEDULE",
    "DEFAULT_TEACHER_REPLAY",
    "LDFWindowPlan",
    "SelfForcingResult",
    "SelfForcingState",
    "resolve_self_forcing_k",
    "run_self_forcing_rollout",
    "sample_rollout_steps",
    "sample_window_plan",
]
