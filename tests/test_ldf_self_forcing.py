from __future__ import annotations

import torch
import torch.nn as nn

from utils.conditions.ldf import HybridMotion, LDFCondition, LDFPrediction
from utils.training.ldf.batch import build_ldf_training_step
from utils.training.ldf.losses import compute_velocity_loss
from utils.training.ldf.flow import (
    build_span_beta,
    recover_clean_for_full_gradient_auxiliary,
    recover_clean_for_self_forcing,
)
from utils.training.ldf.self_forcing import (
    SelfForcingState,
    run_self_forcing_rollout,
    sample_rollout_steps,
    sample_window_plan,
    self_forcing_phase_progress,
)


def empty_condition(batch_size: int):
    def build(_view):
        return LDFCondition(
            text_context=[torch.zeros(1, 1) for _ in range(batch_size)],
            text_null_context=[torch.zeros(1, 1) for _ in range(batch_size)],
        )

    return build


def physical_batch(*, cold: bool = False) -> dict[str, torch.Tensor]:
    root = torch.zeros(2, 40, 5)
    root[..., 0] = torch.arange(40)
    root[..., 2] = torch.arange(40) * 2
    root[..., 3] = 1.0
    return {
        "root_motion": root,
        "source_start_token": torch.zeros(2, dtype=torch.long)
        if cold
        else torch.tensor([1, 3]),
        "cold_start_mask": torch.full((2,), cold, dtype=torch.bool),
    }


def test_window_plan_freezes_geometry_anchor_phase_and_absolute_noise():
    plan = sample_window_plan(
        physical_batch(),
        active_tokens=5,
        rollout_steps=5,
        latent_dim=8,
        initial_history_tokens=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert plan.span_tokens == 10
    assert plan.initial_history_tokens == 1
    assert plan.active_tokens == 5
    assert plan.frontier_tokens == 4
    assert plan.translation_anchor_frame.tolist() == [3, 3]
    assert plan.translation_anchor_xz.tolist() == [[3.0, 6.0], [3.0, 6.0]]
    assert plan.root_noise.shape == (2, 10, 4, 5)
    assert plan.body_noise.shape == (2, 10, 8)
    assert not torch.equal(plan.root_noise[0], plan.root_noise[1])


def test_true_cold_plan_requires_sequence_start_and_uses_frame_zero_anchor():
    plan = sample_window_plan(
        physical_batch(cold=True),
        active_tokens=5,
        rollout_steps=1,
        latent_dim=4,
        generator=torch.Generator().manual_seed(2),
    )

    assert plan.initial_history_tokens == 0
    assert plan.translation_anchor_frame.tolist() == [0, 0]
    assert plan.translation_anchor_xz.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_one_beta_function_defines_history_active_and_frontier():
    phase = torch.tensor([0.05])
    first = build_span_beta(
        span_tokens=10,
        initial_history_tokens=2,
        active_tokens=5,
        phase_offset=phase,
        step_index=0,
    )
    second = build_span_beta(
        span_tokens=10,
        initial_history_tokens=2,
        active_tokens=5,
        phase_offset=phase,
        step_index=1,
    )

    assert torch.allclose(
        first,
        torch.tensor([[0.0, 0.0, 0.15, 0.35, 0.55, 0.75, 0.95, 1.0, 1.0, 1.0]]),
    )
    assert torch.allclose(second[:, 3:7], first[:, 3:7] - 0.2)
    assert second[0, 2] == 0
    assert second[0, 8] == 1


def test_training_steps_reuse_absolute_noise_and_keep_fixed_span_masks():
    clean = HybridMotion(torch.zeros(1, 10, 4, 5), torch.zeros(1, 10, 3))
    root_noise = torch.arange(200, dtype=torch.float32).reshape(1, 10, 4, 5)
    body_noise = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)
    noise = HybridMotion(root_noise, body_noise)
    kwargs = dict(
        clean_motion=clean,
        noise=noise,
        source_start_token=torch.tensor([4]),
        initial_history_tokens=2,
        active_tokens=5,
        phase_offset=torch.tensor([0.05]),
        previous_root_frame=torch.zeros(1, 5),
        previous_root_valid_mask=torch.ones(1, dtype=torch.bool),
        condition_builder=empty_condition(1),
    )
    first = build_ldf_training_step(step_index=0, **kwargs)
    second = build_ldf_training_step(step_index=1, **kwargs)

    assert first.inputs.history_mask.tolist() == [[True, True] + [False] * 8]
    assert first.inputs.generation_mask.tolist() == [
        [False, False] + [True] * 5 + [False] * 3
    ]
    assert first.loss_mask.tolist() == [[False, False] + [True] * 5 + [False] * 3]
    assert second.loss_mask.tolist() == [[False] * 3 + [True] * 5 + [False] * 2]
    assert second.inputs.generation_mask.tolist() == [
        [False] * 3 + [True] * 5 + [False] * 2
    ]
    assert first.inputs.timeline_position_ids.tolist() == [list(range(4, 14))]
    assert first.inputs.rope_position_ids.tolist() == [list(range(-2, 8))]
    # Token 8 is frontier noise in both steps; token 3 follows one fixed noise path.
    assert torch.equal(first.inputs.noisy_motion.root_motion[:, 8], root_noise[:, 8])
    assert torch.equal(second.inputs.noisy_motion.root_motion[:, 8], root_noise[:, 8])
    expected = first.inputs.beta[:, 3, None, None] * root_noise[:, 3]
    assert torch.equal(first.inputs.noisy_motion.root_motion[:, 3], expected)
    expected_next = second.inputs.beta[:, 3, None, None] * root_noise[:, 3]
    assert torch.equal(second.inputs.noisy_motion.root_motion[:, 3], expected_next)


def test_self_forcing_recovery_attenuates_prediction_error_by_beta():
    clean = torch.tensor([2.0])
    noise = torch.tensor([-1.0])
    beta = torch.tensor([0.2])
    target_velocity = clean - noise
    error = torch.tensor([0.5])
    prediction = target_velocity + error
    noisy = (1.0 - beta) * clean + beta * noise

    sf_clean = recover_clean_for_self_forcing(noisy, beta, prediction)
    auxiliary_clean = recover_clean_for_full_gradient_auxiliary(prediction, noise)
    assert torch.allclose(sf_clean - clean, beta * error)
    assert torch.allclose(auxiliary_clean - clean, error)


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls: list[tuple[bool, bool, int]] = []

    def forward(self, inputs):
        self.calls.append(
            (torch.is_grad_enabled(), self.training, inputs.generation_mask.sum().item())
        )
        root_velocity = torch.ones_like(inputs.noisy_motion.root_motion) * self.scale
        body_velocity = torch.ones_like(inputs.noisy_motion.latent_motion) * self.scale
        clean_root = recover_clean_for_self_forcing(
            inputs.noisy_motion.root_motion, inputs.beta, root_velocity
        )
        local = torch.zeros(*clean_root.shape[:-1], 4)
        return LDFPrediction(
            velocity=HybridMotion(root_velocity, body_velocity),
            clean_root_motion=clean_root,
            local_root_motion=local,
            local_root_feature_valid=torch.ones_like(local, dtype=torch.bool),
        )


def test_k_step_rollout_detaches_only_left_boundary_and_backprops_final_step():
    batch = physical_batch()
    plan = sample_window_plan(
        batch,
        active_tokens=5,
        rollout_steps=3,
        latent_dim=3,
        initial_history_tokens=2,
        phase_offset=torch.full((2,), 0.05),
        generator=torch.Generator().manual_seed(11),
    )
    clean = HybridMotion(
        torch.zeros(2, 10, 4, 5),
        torch.zeros(2, 10, 3),
    )
    state = SelfForcingState(clean)
    model = RecordingModel().train()
    views = []

    def condition_builder(view):
        views.append(view)
        return empty_condition(2)(view)

    result = run_self_forcing_rollout(
        model,
        state,
        plan,
        previous_root_frame=torch.zeros(2, 5),
        previous_root_valid_mask=torch.ones(2, dtype=torch.bool),
        condition_builder=condition_builder,
    )

    assert [(grad, training) for grad, training, _ in model.calls] == [
        (False, True),
        (False, True),
        (True, True),
    ]
    assert [view.active_start for view in views] == [2, 3, 4]
    assert len(result.replacements) == 2
    assert all(not item.root_motion.requires_grad for item in result.replacements)
    assert all(item.root_motion.grad_fn is None for item in result.replacements)
    assert not result.state.clean_motion.root_motion.requires_grad
    assert result.state.clean_motion.root_motion[:, :2].equal(clean.root_motion[:, :2])
    assert not result.state.clean_motion.root_motion[:, 2:4].equal(
        clean.root_motion[:, 2:4]
    )
    assert result.state.clean_motion.root_motion[:, 4:].equal(clean.root_motion[:, 4:])

    losses = compute_velocity_loss(result.prediction, result.final_step)
    losses["total"].backward()
    assert model.scale.grad is not None


def test_teacher_replay_is_configurable_without_changing_k_curriculum():
    generator = torch.Generator().manual_seed(0)
    assert sample_rollout_steps(
        0.2, generator=generator, teacher_replay={2: 1.0}
    ) == 1
    assert sample_rollout_steps(
        0.2, generator=generator, teacher_replay={2: 0.0}
    ) == 2
    assert sample_rollout_steps(
        0.5, generator=generator, teacher_replay={3: 0.0}
    ) == 3
    assert sample_rollout_steps(
        0.8, generator=generator, teacher_replay={5: 0.0}
    ) == 5


def test_self_forcing_curriculum_progress_is_relative_to_finetune_phase():
    kwargs = {"phase_start_step": 300_000, "phase_steps": 200_000}
    assert self_forcing_phase_progress(299_999, **kwargs) == 0.0
    assert self_forcing_phase_progress(300_000, **kwargs) == 0.0
    assert self_forcing_phase_progress(400_000, **kwargs) == 0.5
    assert self_forcing_phase_progress(500_000, **kwargs) == 1.0
    assert self_forcing_phase_progress(600_000, **kwargs) == 1.0
