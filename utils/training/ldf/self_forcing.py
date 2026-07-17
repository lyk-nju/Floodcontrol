"""Detached fixed-parent-window self-forcing for LDF training."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
DEFAULT_TEACHER_REPLAY = {2: 0.5, 3: 0.3, 5: 0.2}


def _require_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        converted = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if not math.isfinite(numeric) or numeric != converted:
        raise ValueError(f"{name} must be an integer")
    return converted


def _normalize_k_schedule(schedule) -> tuple[tuple[float, int], ...]:
    rows = []
    for index, row in enumerate(schedule):
        try:
            row_length = len(row)
        except TypeError as error:
            raise ValueError(
                f"self-forcing schedule row {index} must be [threshold,K]"
            ) from error
        if row_length != 2:
            raise ValueError(f"self-forcing schedule row {index} must be [threshold,K]")
        threshold = float(row[0])
        rollout_steps = _require_integer("self-forcing K", row[1])
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("self-forcing thresholds must lie in [0,1]")
        if rollout_steps < 2:
            raise ValueError("self-forcing K values must be at least 2")
        rows.append((threshold, rollout_steps))
    if not rows or rows[0][0] != 0.0:
        raise ValueError("self-forcing schedule must start at threshold 0")
    if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
        raise ValueError(
            "self-forcing thresholds must be strictly increasing and unique"
        )
    if any(right[1] <= left[1] for left, right in zip(rows, rows[1:])):
        raise ValueError("self-forcing K values must be strictly increasing")
    return tuple(rows)


def validate_self_forcing_config(
    config: Mapping[str, object],
    *,
    generation_tokens: int,
    max_window_tokens: int,
    max_steps: int,
) -> tuple[tuple[float, int], ...]:
    """Validate the complete self-forcing phase before training starts."""

    phase_start_step = _require_integer(
        "self_forcing.phase_start_step", config.get("phase_start_step", -1)
    )
    phase_steps = _require_integer(
        "self_forcing.phase_steps", config.get("phase_steps", 0)
    )
    if phase_start_step < 0:
        raise ValueError("self_forcing.phase_start_step must be non-negative")
    if phase_steps <= 0:
        raise ValueError("self_forcing.phase_steps must be positive")

    schedule = _normalize_k_schedule(list(config.get("k_schedule") or []))
    replay_config = dict(config.get("teacher_replay") or {})
    replay: dict[int, float] = {}
    for raw_key, raw_probability in replay_config.items():
        key = _require_integer("self-forcing teacher replay K", raw_key)
        if key in replay:
            raise ValueError("self-forcing teacher replay contains duplicate K keys")
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "self-forcing teacher replay probabilities must lie in [0,1]"
            )
        replay[key] = probability
    schedule_k = {rollout_steps for _, rollout_steps in schedule}
    if set(replay) != schedule_k:
        raise ValueError(
            "self-forcing teacher replay keys must exactly match schedule K values"
        )

    generation_tokens = _require_integer(
        "training.window.generation_tokens", generation_tokens
    )
    max_window_tokens = _require_integer(
        "training.window.max_tokens", max_window_tokens
    )
    if generation_tokens <= 0 or max_window_tokens <= 0:
        raise ValueError("self-forcing window sizes must be positive")
    maximum_rollout = max(rollout_steps for _, rollout_steps in schedule)
    if generation_tokens + maximum_rollout - 1 > max_window_tokens:
        raise ValueError("self-forcing rollout cannot fit inside the training window")

    max_steps = _require_integer("trainer.max_steps", max_steps)
    if max_steps <= 0:
        raise ValueError("trainer.max_steps must be positive")
    if bool(config.get("enabled", False)) and phase_start_step >= max_steps:
        raise ValueError(
            "enabled self-forcing requires phase_start_step < trainer.max_steps"
        )
    return schedule


@dataclass(frozen=True)
class LDFWindowPlan:
    """Immutable geometry, coordinate frame and noise for one rollout."""

    span_tokens: int
    span_token_count: torch.Tensor
    initial_history_tokens: torch.Tensor
    active_tokens: int
    frontier_tokens: torch.Tensor
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

    def validate_structure(self) -> None:
        """Check the rollout tensor contract without reading GPU contents."""

        batch = int(self.root_noise.shape[0])
        if self.span_tokens <= 0 or self.active_tokens <= 0:
            raise ValueError("span_tokens and active_tokens must be positive")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        for name, value, shape in (
            ("span_token_count", self.span_token_count, (batch,)),
            ("initial_history_tokens", self.initial_history_tokens, (batch,)),
            ("frontier_tokens", self.frontier_tokens, (batch,)),
        ):
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if value.dtype != torch.long:
                raise TypeError(f"{name} must be long")
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
        devices = {
            value.device
            for value in (
                self.span_token_count,
                self.initial_history_tokens,
                self.frontier_tokens,
                self.source_start_token,
                self.phase_offset,
                self.translation_anchor_frame,
                self.translation_anchor_xz,
                self.root_noise,
                self.body_noise,
                self.cold_start_mask,
            )
        }
        if len(devices) != 1:
            raise ValueError("all LDFWindowPlan tensors must share a device")
        self.noise.validate()

    def validate(self) -> None:
        """Run the complete rollout geometry and value validation."""

        self.validate_structure()
        if bool((self.span_token_count <= 0).any()) or bool(
            (self.span_token_count > self.span_tokens).any()
        ):
            raise ValueError("real span lengths must lie within the padded span")
        if bool((self.initial_history_tokens < 0).any()) or bool(
            (self.frontier_tokens < 0).any()
        ):
            raise ValueError("history/frontier lengths must be non-negative")
        if not torch.equal(
            self.initial_history_tokens + self.active_tokens + self.frontier_tokens,
            self.span_token_count,
        ):
            raise ValueError("each real span must equal H + active + frontier")
        if bool((self.frontier_tokens < self.rollout_steps - 1).any()):
            raise ValueError("frontier cannot support the requested rollout depth")
        if bool((self.phase_offset < 0).any()) or bool(
            (self.phase_offset >= 1.0 / self.active_tokens).any()
        ):
            raise ValueError("phase_offset must lie in [0,1/active_tokens)")
        if not torch.equal(self.cold_start_mask, self.initial_history_tokens == 0):
            raise ValueError("cold_start_mask must exactly identify H=0 samples")
        if bool((self.cold_start_mask & (self.source_start_token != 0)).any()):
            raise ValueError("H=0 requires a parent at the true sequence start")


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
        if not 0 <= self.completed_steps <= plan.rollout_steps:
            raise ValueError("completed_steps lies outside the rollout")

    def replace_committed_token(
        self,
        *,
        token_index: torch.Tensor,
        root_motion: torch.Tensor,
        latent_motion: torch.Tensor,
    ) -> "SelfForcingState":
        root = self.clean_motion.root_motion.clone()
        latent = self.clean_motion.latent_motion.clone()
        indices = token_index.to(device=root.device, dtype=torch.long).view(-1)
        if tuple(indices.shape) != (root.shape[0],):
            raise ValueError("token_index must be [B]")
        batch_index = torch.arange(root.shape[0], device=root.device)
        root[batch_index, indices] = root_motion.detach()
        latent[batch_index, indices] = latent_motion.detach()
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
    is_rollout: bool = False
    solver_state: object | None = None


def resolve_self_forcing_k(
    progress: float,
    schedule=DEFAULT_K_SCHEDULE,
) -> int:
    """Resolve the current K from monotonically increasing progress thresholds."""

    progress = float(progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0,1]")
    rows = _normalize_k_schedule(schedule)
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
    draw_device = (
        torch.device(generator.device)
        if generator is not None
        else torch.device("cpu")
    )
    draw = float(
        torch.rand((), device=draw_device, generator=generator).item()
    )
    return 1 if draw < replay_probability else rollout_steps


def self_forcing_phase_progress(
    global_step: int,
    *,
    phase_start_step: int,
    phase_steps: int,
) -> float:
    """Measure curriculum progress relative to the self-forcing fine-tune phase."""

    phase_start_step = int(phase_start_step)
    phase_steps = int(phase_steps)
    if phase_start_step < 0:
        raise ValueError("phase_start_step must be non-negative")
    if phase_steps <= 0:
        raise ValueError("phase_steps must be positive")
    return min(
        1.0,
        max(0.0, (int(global_step) - phase_start_step) / float(phase_steps)),
    )


def sample_window_plan(
    batch: dict[str, object],
    *,
    active_tokens: int,
    rollout_steps: int,
    latent_dim: int,
    generator: torch.Generator | None = None,
    initial_history_tokens: int | torch.Tensor | None = None,
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
    span_count = batch["span_token_count"].to(device=root.device, dtype=torch.long)
    context_count = batch["context_token_count"].to(
        device=root.device, dtype=torch.long
    )
    previous_root_valid = batch["previous_root_valid_mask"].to(
        device=root.device, dtype=torch.bool
    )
    if tuple(source_start.shape) != (batch_size,) or tuple(span_count.shape) != (
        batch_size,
    ):
        raise ValueError("source_start_token and span_token_count must be [B]")
    if tuple(context_count.shape) != (batch_size,) or tuple(
        previous_root_valid.shape
    ) != (batch_size,):
        raise ValueError(
            "context_token_count and previous_root_valid_mask must be [B]"
        )
    maximum_history = span_count - active_tokens - (rollout_steps - 1)
    # Persistent solver rollout models steady-state continuation transactions.
    # True cold start has a longer first warm-up transaction and remains covered
    # by the unchanged K=1 ideal path.
    minimum_history = torch.maximum(
        (source_start > 0).to(dtype=torch.long),
        torch.full_like(source_start, 1 if rollout_steps > 1 else 0),
    )
    if initial_history_tokens is None:
        draws = torch.rand(
            batch_size,
            device=root.device,
            generator=generator,
        )
        history = torch.floor(
            draws
            * (maximum_history - minimum_history + 1).to(draws.dtype)
        ).to(torch.long) + minimum_history
    else:
        history = torch.as_tensor(
            initial_history_tokens,
            device=root.device,
            dtype=torch.long,
        ).reshape(-1)
        if history.numel() == 1:
            history = history.expand(batch_size).clone()
        if tuple(history.shape) != (batch_size,):
            raise ValueError("initial_history_tokens must be scalar or [B]")
        # Explicit history is an external validation/evaluation override.  The
        # ordinary training path samples from collator-owned bounds and avoids
        # reading those CUDA values back on the CPU.
        if bool((history < minimum_history).any()):
            if bool(((history == 0) & (source_start > 0)).any()):
                raise ValueError("H=0 requires a parent at the true sequence start")
            raise ValueError("history lies before the valid parent boundary")
        if bool((history > maximum_history).any()):
            raise ValueError("history leaves insufficient rollout frontier")
        cold_override = history == 0
        if bool((cold_override & (context_count != 0)).any()):
            raise ValueError("H=0 requires zero encoder context")
        if bool((cold_override & previous_root_valid).any()):
            raise ValueError("H=0 requires an invalid previous-root boundary")
    frontier = span_count - history - active_tokens
    cold = history == 0

    anchor_frame = torch.where(
        history > 0,
        history * FRAMES_PER_TOKEN - 1,
        torch.zeros_like(history),
    )
    batch_indices = torch.arange(batch_size, device=root.device)
    anchor_xz = root[batch_indices, anchor_frame][:, [0, 2]].clone()

    if phase_offset is None:
        if rollout_steps == 1:
            phase = torch.rand(
                batch_size,
                device=root.device,
                dtype=root.dtype,
                generator=generator,
            ) / float(active_tokens)
        else:
            phase = torch.zeros(
                batch_size, device=root.device, dtype=root.dtype
            )
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
        span_token_count=span_count.clone(),
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
    plan.validate_structure()
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

    plan.validate_structure()
    state.validate(plan)
    if plan.rollout_steps > 1:
        from utils.training.ldf.rollout import run_persistent_rollout

        final_step, prediction, solver_state, replacements = run_persistent_rollout(
            model,
            state.clean_motion,
            plan,
            previous_root_frame=previous_root_frame,
            previous_root_valid_mask=previous_root_valid_mask,
            condition_builder=condition_builder,
        )
        final_clean = solver_state.clean_motion
        return SelfForcingResult(
            final_step=final_step,
            prediction=prediction,
            state=SelfForcingState(
                final_clean,
                completed_steps=plan.rollout_steps,
            ),
            replacements=replacements,
            is_rollout=True,
            solver_state=solver_state,
        )
    replacements: list[HybridMotion] = []
    final_step = None
    final_prediction = None
    current = state

    for step_index in range(plan.rollout_steps):
        training_step = build_ldf_training_step(
            clean_motion=current.clean_motion,
            noise=plan.noise,
            source_start_token=plan.source_start_token,
            span_token_count=plan.span_token_count,
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
        batch_index = torch.arange(
            plan.root_noise.shape[0], device=plan.root_noise.device
        )
        root_replacement = prediction.clean_root_motion[
            batch_index, replace_index
        ].detach()
        body_replacement = recover_clean_for_self_forcing(
            training_step.inputs.noisy_motion.latent_motion[
                batch_index, replace_index
            ],
            training_step.inputs.beta[batch_index, replace_index],
            prediction.velocity.latent_motion[batch_index, replace_index],
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
    "self_forcing_phase_progress",
]
