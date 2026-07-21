"""Ideal, cold-start and persistent solver objectives for LDF training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from utils.conditions.ldf import HybridMotion, LDFCondition, LDFPrediction
from utils.training.ldf.steps import (
    ConditionBuilder,
    LDFStepView,
    LDFTrainingStep,
    build_cold_start_training_step,
    build_ldf_rollout_step,
    build_ldf_training_step,
)
from utils.training.ldf.flow import mix_fixed_noise
from utils.training.ldf.window import LDFWindowPlan


@dataclass(frozen=True)
class LDFSolverResult:
    """One differentiable training endpoint and its detached rollout trace."""

    final_step: LDFTrainingStep
    prediction: LDFPrediction
    clean_motion: HybridMotion
    replacements: tuple[HybridMotion, ...]
    is_rollout: bool = False
    persistent_state: "PersistentRolloutState | None" = None


@dataclass
class PersistentRolloutState:
    """Detached state carried across training commit transactions."""

    noisy_motion: HybridMotion
    clean_motion: HybridMotion
    beta: torch.Tensor
    current_denoise_step: torch.Tensor
    previous_root_frame: torch.Tensor | None
    previous_root_valid_mask: torch.Tensor | None
    origin_xz: torch.Tensor
    completed_commits: int = 0


def _scheduler_beta(
    *,
    model,
    token_length: int,
    current_step: torch.Tensor,
    noise_steps: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(
        int(token_length), device=current_step.device, dtype=torch.long
    )[None].expand(current_step.shape[0], -1)
    return model.triangular_beta(
        timeline_position_ids=positions,
        diffusion_time=current_step.to(torch.float32) / float(noise_steps),
    ).to(dtype=dtype)


def _make_view(
    *,
    source_start_token: torch.Tensor,
    history_end: torch.Tensor,
    active_tokens: int,
    token_length: int,
    beta: torch.Tensor,
    commit_offset: int,
) -> LDFStepView:
    positions = torch.arange(
        int(token_length), device=beta.device, dtype=torch.long
    )[None].expand(beta.shape[0], -1)
    active_end = history_end + int(active_tokens)
    return LDFStepView(
        step_index=int(commit_offset),
        history_end=history_end,
        active_start=history_end,
        active_end=active_end,
        frontier_start=active_end,
        timeline_position_ids=source_start_token[:, None] + positions,
        rope_position_ids=positions - history_end[:, None],
        beta=beta,
    )


def _replace_token(
    motion: HybridMotion,
    token_index: torch.Tensor,
    replacement: HybridMotion,
) -> HybridMotion:
    root = motion.root_motion.clone()
    latent = motion.latent_motion.clone()
    batch_index = torch.arange(root.shape[0], device=root.device)
    root[batch_index, token_index] = replacement.root_motion[:, 0]
    latent[batch_index, token_index] = replacement.latent_motion[:, 0]
    return HybridMotion(root, latent)


def _commit_and_rebase(
    model,
    state: PersistentRolloutState,
    *,
    token_index: torch.Tensor,
    next_motion: HybridMotion,
    next_beta: torch.Tensor,
) -> tuple[PersistentRolloutState, HybridMotion]:
    """Commit one predicted token, then shift the model coordinate origin."""

    rebased_motion, committed, translation = model.commit_step(
        next_motion.clone(detach=True),
        next_beta,
        token_index,
    )
    committed = committed.clone(detach=True)

    clean = _replace_token(state.clean_motion, token_index, committed)
    translation = translation.detach()
    clean = model.rebase_motion_state(
        clean.clone(detach=True), torch.zeros_like(next_beta), translation
    )

    previous = state.previous_root_frame
    if previous is not None:
        previous = previous.detach().clone()
        previous[..., [0, 2]] -= translation.to(previous)

    return (
        PersistentRolloutState(
            noisy_motion=HybridMotion(
                rebased_motion.root_motion,
                rebased_motion.latent_motion,
            ),
            clean_motion=clean,
            beta=next_beta.detach(),
            current_denoise_step=state.current_denoise_step,
            previous_root_frame=previous,
            previous_root_valid_mask=state.previous_root_valid_mask,
            origin_xz=state.origin_xz + translation.to(state.origin_xz),
            completed_commits=state.completed_commits + 1,
        ),
        committed,
    )


def run_persistent_rollout(
    model,
    clean_motion: HybridMotion,
    plan,
    *,
    previous_root_frame: torch.Tensor | None,
    previous_root_valid_mask: torch.Tensor | None,
    condition_builder: ConditionBuilder,
    supervised_microstep: int | None = None,
):
    """Evolve a steady or true-cold persistent solver state.

    Conditions are compiled once per commit.  Steady-state rollouts supervise
    the final update as before.  A true-cold rollout starts at diffusion step
    zero and may supervise any microstep in its first two-commit lifecycle;
    preceding updates run under ``no_grad`` and are never reconstructed from an
    ideal bridge.
    """

    if plan.rollout_steps <= 0:
        raise ValueError("persistent rollout requires at least one commit")
    noise_steps = int(model.noise_steps)
    active_tokens = int(plan.active_tokens)
    if active_tokens != int(model.chunk_size):
        raise ValueError("rollout active_tokens must equal the model chunk_size")
    if noise_steps <= 0 or noise_steps % active_tokens:
        raise ValueError(
            "persistent rollout currently requires noise_steps divisible by active_tokens"
        )
    if bool((plan.phase_offset != 0).any()):
        raise ValueError("persistent rollout must start at a runtime commit boundary")

    device = clean_motion.root_motion.device
    history = plan.initial_history_tokens.to(device=device, dtype=torch.long)
    cold_mask = history == 0
    if bool(cold_mask.any()) and not bool(cold_mask.all()):
        raise ValueError("persistent rollout cannot mix cold and steady samples")
    is_cold = bool(cold_mask.all())
    if is_cold:
        if bool((plan.source_start_token != 0).any()):
            raise ValueError("persistent cold rollout requires the true sequence start")
        if previous_root_valid_mask is not None and bool(
            previous_root_valid_mask.any()
        ):
            raise ValueError("persistent cold rollout requires no previous root")
    elif supervised_microstep is not None:
        raise ValueError("supervised_microstep is only valid for persistent cold")
    elif plan.rollout_steps <= 1:
        raise ValueError("steady persistent rollout requires K > 1")

    steps_per_commit = noise_steps // active_tokens
    microsteps_per_commit = [steps_per_commit] * int(plan.rollout_steps)
    if is_cold:
        current_step = torch.zeros_like(history)
        microsteps_per_commit[0] = noise_steps
    else:
        current_step = (history - 1 + active_tokens) * steps_per_commit
    total_microsteps = sum(microsteps_per_commit)
    if supervised_microstep is None:
        differentiable_microstep = total_microsteps - 1
    else:
        differentiable_microstep = int(supervised_microstep)
        if not 0 <= differentiable_microstep < total_microsteps:
            raise ValueError(
                "persistent cold supervised_microstep lies outside its lifecycle"
            )
    beta = _scheduler_beta(
        model=model,
        token_length=clean_motion.token_length,
        current_step=current_step,
        noise_steps=noise_steps,
        dtype=clean_motion.root_motion.dtype,
    )
    state = PersistentRolloutState(
        noisy_motion=mix_fixed_noise(clean_motion, plan.noise, beta).clone(detach=True),
        clean_motion=clean_motion.clone(detach=True),
        beta=beta,
        current_denoise_step=current_step,
        previous_root_frame=(
            None if previous_root_frame is None else previous_root_frame.detach().clone()
        ),
        previous_root_valid_mask=(
            None
            if previous_root_valid_mask is None
            else previous_root_valid_mask.detach().clone()
        ),
        origin_xz=torch.zeros(
            clean_motion.batch_size,
            2,
            device=device,
            dtype=clean_motion.root_motion.dtype,
        ),
    )
    replacements: list[HybridMotion] = []
    final_step: LDFTrainingStep | None = None
    final_prediction = None
    flat_microstep = 0
    stop_after_step = False

    for commit_offset, commit_microsteps in enumerate(microsteps_per_commit):
        history_end = history + int(commit_offset)
        view = _make_view(
            source_start_token=plan.source_start_token.to(device=device),
            history_end=history_end,
            active_tokens=active_tokens,
            token_length=clean_motion.token_length,
            beta=state.beta,
            commit_offset=commit_offset,
        )
        condition = condition_builder(view, state.clean_motion)
        if not isinstance(condition, LDFCondition):
            raise TypeError("condition_builder must return LDFCondition")

        for denoise_offset in range(commit_microsteps):
            next_step = state.current_denoise_step + 1
            next_beta = _scheduler_beta(
                model=model,
                token_length=clean_motion.token_length,
                current_step=next_step,
                noise_steps=noise_steps,
                dtype=clean_motion.root_motion.dtype,
            )
            training_step = build_ldf_rollout_step(
                model=model,
                noisy_motion=state.noisy_motion,
                clean_motion=state.clean_motion,
                noise=plan.noise,
                beta=state.beta,
                next_beta=next_beta,
                source_start_token=plan.source_start_token,
                span_token_count=plan.span_token_count,
                history_end=history_end,
                active_tokens=active_tokens,
                step_index=commit_offset,
                previous_root_frame=state.previous_root_frame,
                previous_root_valid_mask=state.previous_root_valid_mask,
                condition=condition,
            )
            is_supervised = flat_microstep == differentiable_microstep
            if is_supervised:
                next_motion, prediction = model.denoise_step(
                    training_step.inputs,
                    next_beta,
                    use_cfg=False,
                )
                final_step = training_step
                final_prediction = prediction
            else:
                with torch.no_grad():
                    next_motion, _ = model.denoise_step(
                        training_step.inputs,
                        next_beta,
                        use_cfg=False,
                    )
            state.noisy_motion = next_motion
            state.beta = next_beta
            state.current_denoise_step = next_step
            flat_microstep += 1
            if is_supervised:
                stop_after_step = True
                break

        reached_commit_boundary = denoise_offset == commit_microsteps - 1
        if reached_commit_boundary:
            commit_index = history_end
            state, committed = _commit_and_rebase(
                model,
                state,
                token_index=commit_index,
                next_motion=state.noisy_motion,
                next_beta=state.beta,
            )
            replacements.append(committed)
        if stop_after_step:
            break

    if final_step is None or final_prediction is None:
        raise RuntimeError("persistent rollout did not produce a differentiable final step")
    return final_step, final_prediction, state, tuple(replacements)


def run_training_solver(
    model: nn.Module,
    clean_motion: HybridMotion,
    plan: LDFWindowPlan,
    *,
    previous_root_frame: torch.Tensor | None,
    previous_root_valid_mask: torch.Tensor | None,
    condition_builder: ConditionBuilder,
    cold_denoise_step: torch.Tensor | None = None,
    cold_persistent_microstep: int | None = None,
) -> LDFSolverResult:
    """Run the canonical K=1, cold-start or persistent K>1 objective."""

    plan.validate_structure()
    clean_motion.validate()
    if clean_motion.token_length != plan.span_tokens:
        raise ValueError("clean motion does not match the rollout span")
    if clean_motion.batch_size != plan.root_noise.shape[0]:
        raise ValueError("clean motion does not match the rollout batch")

    if cold_denoise_step is not None and cold_persistent_microstep is not None:
        raise ValueError("cold objective cannot be both ideal and persistent")

    if cold_denoise_step is not None:
        if plan.rollout_steps != 1 or not bool(plan.cold_start_mask.all()):
            raise ValueError("cold denoise phases require a true-cold K=1 plan")
        training_step = build_cold_start_training_step(
            model=model,
            clean_motion=clean_motion,
            noise=plan.noise,
            source_start_token=plan.source_start_token,
            span_token_count=plan.span_token_count,
            active_tokens=plan.active_tokens,
            denoise_step_index=cold_denoise_step,
            previous_root_frame=previous_root_frame,
            previous_root_valid_mask=previous_root_valid_mask,
            condition_builder=condition_builder,
        )
        return LDFSolverResult(
            final_step=training_step,
            prediction=model(training_step.inputs),
            clean_motion=clean_motion,
            replacements=(),
        )

    if cold_persistent_microstep is not None:
        if plan.rollout_steps < 1 or not bool(plan.cold_start_mask.all()):
            raise ValueError("persistent cold requires an all-H=0 plan")
        final_step, prediction, state, replacements = run_persistent_rollout(
            model,
            clean_motion,
            plan,
            previous_root_frame=previous_root_frame,
            previous_root_valid_mask=previous_root_valid_mask,
            condition_builder=condition_builder,
            supervised_microstep=cold_persistent_microstep,
        )
        return LDFSolverResult(
            final_step=final_step,
            prediction=prediction,
            clean_motion=state.clean_motion,
            replacements=replacements,
            is_rollout=True,
            persistent_state=state,
        )

    if plan.rollout_steps > 1:
        final_step, prediction, state, replacements = run_persistent_rollout(
            model,
            clean_motion,
            plan,
            previous_root_frame=previous_root_frame,
            previous_root_valid_mask=previous_root_valid_mask,
            condition_builder=condition_builder,
        )
        return LDFSolverResult(
            final_step=final_step,
            prediction=prediction,
            clean_motion=state.clean_motion,
            replacements=replacements,
            is_rollout=True,
            persistent_state=state,
        )

    training_step = build_ldf_training_step(
        model=model,
        clean_motion=clean_motion,
        noise=plan.noise,
        source_start_token=plan.source_start_token,
        span_token_count=plan.span_token_count,
        initial_history_tokens=plan.initial_history_tokens,
        active_tokens=plan.active_tokens,
        phase_offset=plan.phase_offset,
        step_index=0,
        previous_root_frame=previous_root_frame,
        previous_root_valid_mask=previous_root_valid_mask,
        condition_builder=condition_builder,
    )
    return LDFSolverResult(
        final_step=training_step,
        prediction=model(training_step.inputs),
        clean_motion=clean_motion,
        replacements=(),
    )


__all__ = [
    "LDFSolverResult",
    "PersistentRolloutState",
    "run_persistent_rollout",
    "run_training_solver",
]
