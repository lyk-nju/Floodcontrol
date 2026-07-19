from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.diffusion_forcing_wan import LDF
from utils.conditions.ldf import HybridMotion, LDFCondition, LDFInput, LDFPrediction
from utils.training.ldf.steps import (
    LDFStepView,
    build_cold_start_training_step,
    build_ldf_rollout_step,
    build_ldf_training_step,
)
from utils.training.ldf.conditioning import (
    create_xz_condition,
    sample_xz_constraint_mask,
)
from utils.training.ldf.losses import compute_offpath_loss, compute_velocity_loss
from utils.training.ldf.flow import (
    build_span_beta,
    recover_clean_for_full_gradient_auxiliary,
    recover_clean_for_self_forcing,
)
from utils.training.ldf.solver import run_training_solver
from utils.training.ldf.window import (
    resolve_self_forcing_k,
    sample_rollout_steps,
    sample_window_plan,
)


def empty_condition(batch_size: int):
    def build(view, _clean_motion=None):
        token_length = int(view.timeline_position_ids.shape[1])
        null = [torch.zeros(1, 1) for _ in range(batch_size)]
        return LDFCondition(
            text_context=[
                null[batch_index]
                for batch_index in range(batch_size)
                for _ in range(token_length)
            ],
            text_null_context=null,
        )

    return build


def physical_batch(*, cold: bool = False, frames: int = 40) -> dict[str, torch.Tensor]:
    root = torch.zeros(2, frames, 5)
    root[..., 0] = torch.arange(frames)
    root[..., 2] = torch.arange(frames) * 2
    root[..., 3] = 1.0
    source_start = (
        torch.zeros(2, dtype=torch.long)
        if cold
        else torch.tensor([1, 3])
    )
    return {
        "root_motion": root,
        "source_start_token": source_start,
        "span_token_count": torch.full((2,), frames // 4, dtype=torch.long),
        "context_token_count": source_start.clamp_max(2),
        "previous_root_valid_mask": source_start > 0,
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
    assert plan.span_token_count.tolist() == [10, 10]
    assert plan.initial_history_tokens.tolist() == [1, 1]
    assert plan.active_tokens == 5
    assert plan.frontier_tokens.tolist() == [4, 4]
    assert plan.translation_anchor_frame.tolist() == [3, 3]
    assert plan.translation_anchor_xz.tolist() == [[3.0, 6.0], [3.0, 6.0]]
    assert plan.root_noise.shape == (2, 10, 4, 5)
    assert plan.body_noise.shape == (2, 10, 8)
    assert not torch.equal(plan.root_noise[0], plan.root_noise[1])


def test_window_plan_defers_phase_value_checks_to_full_validation():
    plan = sample_window_plan(
        physical_batch(),
        active_tokens=5,
        rollout_steps=1,
        latent_dim=8,
        initial_history_tokens=1,
        phase_offset=torch.full((2,), 0.25),
    )

    plan.validate_structure()
    with pytest.raises(ValueError, match="phase_offset"):
        plan.validate()


def test_true_cold_plan_requires_sequence_start_and_uses_frame_zero_anchor():
    plan = sample_window_plan(
        physical_batch(cold=True),
        active_tokens=5,
        rollout_steps=1,
        latent_dim=4,
        initial_history_tokens=0,
        generator=torch.Generator().manual_seed(2),
    )

    assert plan.initial_history_tokens.tolist() == [0, 0]
    assert plan.cold_start_mask.tolist() == [True, True]
    assert plan.translation_anchor_frame.tolist() == [0, 0]
    assert plan.translation_anchor_xz.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_explicit_cold_replay_contract_removes_accidental_h_zero_sampling():
    histories = []
    for seed in range(20):
        plan = sample_window_plan(
            physical_batch(cold=True),
            active_tokens=5,
            rollout_steps=1,
            latent_dim=4,
            allow_cold_start=False,
            generator=torch.Generator().manual_seed(seed),
        )
        histories.extend(plan.initial_history_tokens.tolist())

    assert min(histories) >= 1


def test_h_zero_rejects_mid_clip_context_and_previous_root_boundaries():
    with pytest.raises(ValueError, match="true sequence start"):
        sample_window_plan(
            physical_batch(),
            active_tokens=5,
            rollout_steps=1,
            latent_dim=4,
            initial_history_tokens=0,
        )

    hidden_context = physical_batch(cold=True)
    hidden_context["context_token_count"][0] = 1
    with pytest.raises(ValueError, match="zero encoder context"):
        sample_window_plan(
            hidden_context,
            active_tokens=5,
            rollout_steps=1,
            latent_dim=4,
            initial_history_tokens=0,
        )

    hidden_boundary = physical_batch(cold=True)
    hidden_boundary["previous_root_valid_mask"][0] = True
    with pytest.raises(ValueError, match="invalid previous-root"):
        sample_window_plan(
            hidden_boundary,
            active_tokens=5,
            rollout_steps=1,
            latent_dim=4,
            initial_history_tokens=0,
        )


def test_window_plan_samples_per_sample_history_and_reserves_rollout_frontier():
    histories = []
    for seed in range(20):
        plan = sample_window_plan(
            physical_batch(frames=120),
            active_tokens=5,
            rollout_steps=1,
            latent_dim=4,
            generator=torch.Generator().manual_seed(seed),
        )
        histories.extend(plan.initial_history_tokens.tolist())
    assert min(histories) == 1
    assert max(histories) <= 25
    assert len(set(histories)) > 10

    plan = sample_window_plan(
        physical_batch(frames=120),
        active_tokens=5,
        rollout_steps=5,
        latent_dim=4,
        initial_history_tokens=torch.tensor([20, 21]),
    )
    assert plan.frontier_tokens.tolist() == [5, 4]
    with pytest.raises(ValueError, match="insufficient rollout frontier"):
        sample_window_plan(
            physical_batch(frames=120),
            active_tokens=5,
            rollout_steps=5,
            latent_dim=4,
            initial_history_tokens=torch.tensor([22, 21]),
        )


def test_one_beta_function_defines_history_active_and_frontier():
    phase = torch.tensor([0.05])
    first = build_span_beta(
        span_tokens=10,
        initial_history_tokens=torch.tensor([2]),
        active_tokens=5,
        phase_offset=phase,
        step_index=0,
    )
    second = build_span_beta(
        span_tokens=10,
        initial_history_tokens=torch.tensor([2]),
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
        model=RecordingModel(),
        clean_motion=clean,
        noise=noise,
        source_start_token=torch.tensor([4]),
        span_token_count=torch.tensor([10]),
        initial_history_tokens=torch.tensor([2]),
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


@pytest.mark.parametrize("denoise_step", range(10))
def test_cold_ideal_phase_matches_runtime_beta_visibility_and_state(denoise_step):
    model = LDF(
        latent_dim=3,
        root_mean=[0.0] * 5,
        root_std=[1.0] * 5,
        local_root_mean=[0.0] * 4,
        local_root_std=[1.0] * 4,
        hidden_dim=16,
        ffn_dim=32,
        freq_dim=8,
        text_dim=1,
        text_len=1,
        num_heads=4,
        root_num_layers=1,
        body_num_layers=1,
        chunk_size=5,
        noise_steps=10,
    )
    clean = HybridMotion(
        torch.zeros(1, 10, 4, 5),
        torch.zeros(1, 10, 3),
    )
    noise = HybridMotion(
        torch.ones_like(clean.root_motion),
        torch.ones_like(clean.latent_motion),
    )
    step = build_cold_start_training_step(
        model=model,
        clean_motion=clean,
        noise=noise,
        source_start_token=torch.zeros(1, dtype=torch.long),
        span_token_count=torch.tensor([10]),
        active_tokens=5,
        denoise_step_index=torch.tensor([denoise_step]),
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition_builder=empty_condition(1),
    )

    positions = torch.arange(10)[None]
    beta = model.triangular_beta(
        timeline_position_ids=positions,
        diffusion_time=denoise_step / 10.0,
    )
    next_beta = model.triangular_beta(
        timeline_position_ids=positions,
        diffusion_time=(denoise_step + 1) / 10.0,
    )
    runtime_inputs = model.create_input(
        HybridMotion(
            beta[..., None, None] * noise.root_motion,
            beta[..., None] * noise.latent_motion,
        ),
        beta=beta,
        next_beta=next_beta,
        timeline_position_ids=positions,
        commit_index=0,
        condition=step.inputs.condition,
        previous_root_frame=None,
        previous_root_valid_mask=None,
    )

    assert torch.equal(step.inputs.beta, runtime_inputs.beta)
    assert torch.equal(
        step.inputs.generation_mask,
        runtime_inputs.generation_mask,
    )
    assert torch.equal(step.loss_mask, runtime_inputs.generation_mask)
    assert torch.equal(
        step.inputs.noisy_motion.root_motion,
        runtime_inputs.noisy_motion.root_motion,
    )
    assert torch.equal(
        step.inputs.noisy_motion.latent_motion,
        runtime_inputs.noisy_motion.latent_motion,
    )
    assert torch.equal(step.inputs.timeline_position_ids, positions)
    assert torch.equal(step.inputs.rope_position_ids, positions)
    assert int(step.loss_mask.sum()) == 1 + denoise_step // 2


def test_cold_dynamic_future_starts_at_each_microstep_visible_end():
    model = LDF(
        latent_dim=3,
        root_mean=[0.0] * 5,
        root_std=[1.0] * 5,
        local_root_mean=[0.0] * 4,
        local_root_std=[1.0] * 4,
        hidden_dim=16,
        ffn_dim=32,
        freq_dim=8,
        text_dim=1,
        text_len=1,
        num_heads=4,
        root_num_layers=1,
        body_num_layers=1,
        chunk_size=5,
        noise_steps=10,
    )
    clean = HybridMotion(
        torch.zeros(1, 10, 4, 5),
        torch.zeros(1, 10, 3),
    )
    noise = HybridMotion(
        torch.ones_like(clean.root_motion),
        torch.ones_like(clean.latent_motion),
    )
    constraint_mask = torch.zeros_like(clean.root_motion, dtype=torch.bool)
    constraint_mask[:, :8, :, 0] = True
    constraint_mask[:, :8, :, 2] = True

    def condition_builder(view, clean_motion):
        return create_xz_condition(
            clean_root_motion=clean_motion.root_motion,
            token_valid_mask=torch.ones(1, 10, dtype=torch.bool),
            constraint_mask=constraint_mask,
            view=view,
            text_context=[torch.zeros(1, 1) for _ in range(10)],
            text_null_context=[torch.zeros(1, 1)],
            future_horizon_tokens=torch.tensor([3]),
        )

    visible_counts = []
    selected_positions = []
    candidate_positions = []
    for denoise_step in range(10):
        step = build_cold_start_training_step(
            model=model,
            clean_motion=clean,
            noise=noise,
            source_start_token=torch.zeros(1, dtype=torch.long),
            span_token_count=torch.tensor([10]),
            active_tokens=5,
            denoise_step_index=torch.tensor([denoise_step]),
            previous_root_frame=None,
            previous_root_valid_mask=None,
            condition_builder=condition_builder,
        )
        visible = int(step.inputs.generation_mask.sum().item())
        selected = step.inputs.future_attention_mask()[0]
        positions = step.inputs.condition.future_timeline_position_ids[0]
        visible_counts.append(visible)
        candidate_positions.append(positions.tolist())
        selected_positions.append(positions[selected].tolist())
        assert all(position >= visible for position in positions[selected].tolist())

    assert visible_counts == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert all(positions == list(range(1, 8)) for positions in candidate_positions)
    assert selected_positions == [
        [1, 2, 3],
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 4],
        [3, 4, 5],
        [3, 4, 5],
        [4, 5, 6],
        [4, 5, 6],
        [5, 6, 7],
        [5, 6, 7],
    ]


def test_mixed_batch_uses_each_samples_own_history_and_real_span_length():
    clean = HybridMotion(
        torch.zeros(2, 10, 4, 5),
        torch.zeros(2, 10, 3),
    )
    noise = HybridMotion(
        torch.ones_like(clean.root_motion),
        torch.ones_like(clean.latent_motion),
    )
    step = build_ldf_training_step(
        model=RecordingModel(),
        clean_motion=clean,
        noise=noise,
        source_start_token=torch.tensor([10, 20]),
        span_token_count=torch.tensor([8, 10]),
        initial_history_tokens=torch.tensor([0, 5]),
        active_tokens=5,
        phase_offset=torch.tensor([0.05, 0.05]),
        step_index=0,
        previous_root_frame=torch.zeros(2, 5),
        previous_root_valid_mask=torch.ones(2, dtype=torch.bool),
        condition_builder=empty_condition(2),
    )

    assert step.inputs.history_mask.tolist() == [
        [False] * 10,
        [True] * 5 + [False] * 5,
    ]
    assert step.inputs.generation_mask.tolist() == [
        [True] * 5 + [False] * 5,
        [False] * 5 + [True] * 5,
    ]
    assert step.loss_mask.tolist() == step.inputs.generation_mask.tolist()
    assert step.inputs.rope_position_ids.tolist() == [
        list(range(10)),
        list(range(-5, 5)),
    ]
    assert step.inputs.timeline_position_ids.tolist() == [
        list(range(10, 20)),
        list(range(20, 30)),
    ]


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
        self.noisy_inputs: list[HybridMotion] = []
        self.beta_inputs: list[torch.Tensor] = []
        self.noise_steps = 10
        self.chunk_size = 5
        self.register_buffer("root_mean", torch.zeros(5))
        self.register_buffer("root_std", torch.ones(5))

    def denormalize_root(self, root):
        return root

    def _project_normalized_root(self, root):
        return root

    def triangular_beta(self, *, timeline_position_ids, diffusion_time):
        time = torch.as_tensor(diffusion_time, dtype=torch.float32)
        if time.ndim == 1:
            time = time[:, None]
        return torch.clamp(
            1.0 + timeline_position_ids.float() / 5.0 - time,
            min=0.0,
            max=1.0,
        )

    def rebase_motion_state(self, motion, beta, translation_xz):
        root = motion.root_motion.clone()
        root[..., [0, 2]] -= (
            (1.0 - beta)[..., None, None]
            * translation_xz[:, None, None, :]
        )
        return HybridMotion(root, motion.latent_motion.clone())

    def create_input(self, *args, **kwargs):
        return LDF.create_input(self, *args, **kwargs)

    def commit_step(self, *args, **kwargs):
        return LDF.commit_step(self, *args, **kwargs)

    def forward(self, inputs):
        self.calls.append(
            (torch.is_grad_enabled(), self.training, inputs.generation_mask.sum().item())
        )
        self.noisy_inputs.append(inputs.noisy_motion.clone(detach=True))
        self.beta_inputs.append(inputs.beta.detach().clone())
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

    def denoise_step(self, inputs, next_beta, *, use_cfg):
        assert not use_cfg
        prediction = self(inputs)
        delta = inputs.beta - next_beta
        return HybridMotion(
            inputs.noisy_motion.root_motion
            + delta[..., None, None] * prediction.velocity.root_motion,
            inputs.noisy_motion.latent_motion
            + delta[..., None] * prediction.velocity.latent_motion,
        ), prediction


def test_k_step_rollout_detaches_only_left_boundary_and_backprops_final_step(
    monkeypatch,
):
    import utils.training.ldf.solver as solver_module

    mix_calls = 0
    original_mix = solver_module.mix_fixed_noise

    def counted_mix(*args, **kwargs):
        nonlocal mix_calls
        mix_calls += 1
        return original_mix(*args, **kwargs)

    monkeypatch.setattr(solver_module, "mix_fixed_noise", counted_mix)
    batch = physical_batch()
    plan = sample_window_plan(
        batch,
        active_tokens=5,
        rollout_steps=3,
        latent_dim=3,
        initial_history_tokens=2,
        phase_offset=torch.zeros(2),
        generator=torch.Generator().manual_seed(11),
    )
    clean = HybridMotion(
        torch.zeros(2, 10, 4, 5),
        torch.zeros(2, 10, 3),
    )
    model = RecordingModel().train()
    views = []

    def condition_builder(view, _clean_motion):
        views.append(view)
        return empty_condition(2)(view)

    result = run_training_solver(
        model,
        clean,
        plan,
        previous_root_frame=torch.zeros(2, 5),
        previous_root_valid_mask=torch.ones(2, dtype=torch.bool),
        condition_builder=condition_builder,
    )

    assert [(grad, training) for grad, training, _ in model.calls] == [
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (True, True),
    ]
    assert mix_calls == 1
    first_delta = model.beta_inputs[0] - model.beta_inputs[1]
    assert torch.allclose(
        model.noisy_inputs[1].root_motion,
        model.noisy_inputs[0].root_motion
        + first_delta[..., None, None] * 0.25,
    )
    assert torch.allclose(
        model.noisy_inputs[1].latent_motion,
        model.noisy_inputs[0].latent_motion + first_delta[..., None] * 0.25,
    )
    assert [view.active_start.tolist() for view in views] == [[2, 2], [3, 3], [4, 4]]
    assert len(result.replacements) == 3
    assert all(not item.root_motion.requires_grad for item in result.replacements)
    assert all(item.root_motion.grad_fn is None for item in result.replacements)
    assert not result.clean_motion.root_motion.requires_grad
    # Root history is expressed in the new per-commit coordinate origin, while
    # body latent history is translation invariant.
    assert not result.clean_motion.root_motion[:, :2].equal(
        clean.root_motion[:, :2]
    )
    assert result.clean_motion.latent_motion[:, :2].equal(
        clean.latent_motion[:, :2]
    )
    assert result.persistent_state is not None
    assert result.persistent_state.completed_commits == 3
    assert not result.clean_motion.root_motion[:, 2:4].equal(
        clean.root_motion[:, 2:4]
    )

    from utils.training.ldf.losses import compute_offpath_loss

    losses = compute_offpath_loss(
        result.prediction,
        result.final_step,
        root_mean=model.root_mean,
        root_std=model.root_std,
    )
    ideal_losses = compute_velocity_loss(
        result.prediction,
        type(
            "IdealStep",
            (),
            {
                "loss_mask": result.final_step.loss_mask,
                "target_velocity": result.prediction.velocity,
            },
        )(),
    )
    assert tuple(losses) == tuple(ideal_losses)
    losses["total"].backward()
    assert model.scale.grad is not None


def test_offpath_loss_is_finite_near_zero_beta_and_boundary_loss_backpropagates():
    clean_root = torch.zeros(1, 2, 4, 5)
    clean_root[:, 1, :, 0] = torch.arange(1, 5, dtype=torch.float32)
    clean = HybridMotion(clean_root, torch.zeros(1, 2, 3))
    current = HybridMotion(
        torch.zeros_like(clean.root_motion),
        torch.zeros_like(clean.latent_motion),
    )
    noise = HybridMotion(
        torch.zeros_like(clean.root_motion),
        torch.zeros_like(clean.latent_motion),
    )
    beta = torch.tensor([[0.0, 1.0e-8]])
    step = build_ldf_rollout_step(
        model=RecordingModel(),
        noisy_motion=current,
        clean_motion=clean,
        noise=noise,
        beta=beta,
        next_beta=beta,
        source_start_token=torch.zeros(1, dtype=torch.long),
        span_token_count=torch.tensor([2]),
        history_end=torch.tensor([1]),
        active_tokens=1,
        step_index=0,
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=empty_condition(1)(
            type("View", (), {"timeline_position_ids": torch.zeros(1, 2)})()
        ),
    )
    root_velocity = torch.zeros_like(clean.root_motion, requires_grad=True)
    body_velocity = torch.zeros_like(clean.latent_motion, requires_grad=True)
    local = torch.zeros(1, 2, 4, 4)
    prediction = LDFPrediction(
        HybridMotion(root_velocity, body_velocity),
        clean.root_motion,
        local,
        torch.ones_like(local, dtype=torch.bool),
    )
    losses = compute_offpath_loss(
        prediction,
        step,
        root_mean=torch.zeros(5),
        root_std=torch.ones(5),
        rollout_weight=0.0,
        root_boundary_weight=1.0,
        offpath_beta_min=0.1,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert root_velocity.grad is not None
    assert torch.isfinite(root_velocity.grad).all()
    assert root_velocity.grad.abs().sum() > 0


def test_xz_condition_moves_active_and_future_ranges_with_self_forcing_steps():
    root = torch.zeros(1, 10, 4, 5)
    root[..., 0] = torch.arange(10)[None, :, None]
    root[..., 2] = root[..., 0] * 2
    valid = torch.ones(1, 10, dtype=torch.bool)
    text = [torch.zeros(1, 4) for _ in range(10)]
    null = [torch.zeros(1, 4)]
    conditions = []
    constraint_mask = sample_xz_constraint_mask(
        token_valid_mask=valid,
        initial_active_start=torch.tensor([2]),
        initial_active_end=torch.tensor([7]),
        future_horizon_tokens=torch.tensor([2]),
        rollout_steps=2,
        dense_probability=1.0,
        waypoint_probability=0.0,
        goal_probability=0.0,
        max_waypoint_count=4,
    )
    kwargs = dict(
        clean_root_motion=root,
        token_valid_mask=valid,
        constraint_mask=constraint_mask,
        text_context=text,
        text_null_context=null,
        future_horizon_tokens=torch.tensor([2]),
    )

    def builder(view, clean_motion):
        condition = create_xz_condition(
            view=view,
            **{**kwargs, "clean_root_motion": clean_motion.root_motion},
        )
        conditions.append(condition)
        return condition

    clean = HybridMotion(root, torch.zeros(1, 10, 3))
    plan = sample_window_plan(
        {
            "root_motion": root.flatten(1, 2),
            "source_start_token": torch.tensor([4]),
            "span_token_count": torch.tensor([10]),
            "context_token_count": torch.tensor([2]),
            "previous_root_valid_mask": torch.tensor([True]),
        },
        active_tokens=5,
        rollout_steps=2,
        latent_dim=3,
        initial_history_tokens=2,
        phase_offset=torch.zeros(1),
        generator=torch.Generator().manual_seed(13),
    )
    run_training_solver(
        RecordingModel(),
        clean,
        plan,
        previous_root_frame=torch.zeros(1, 5),
        previous_root_valid_mask=torch.ones(1, dtype=torch.bool),
        condition_builder=builder,
    )

    assert len(conditions) == 2
    first_active = conditions[0].root_condition_mask.flatten(2).any(-1)
    second_active = conditions[1].root_condition_mask.flatten(2).any(-1)
    assert first_active.tolist() == [[False, False] + [True] * 5 + [False] * 3]
    assert second_active.tolist() == [[False] * 3 + [True] * 5 + [False] * 2]
    assert conditions[0].future_timeline_position_ids.tolist() == [
        [7, 8, 9, 10, 11, 12]
    ]
    assert conditions[1].future_timeline_position_ids.tolist() == [
        [8, 9, 10, 11, 12, 13]
    ]


def test_k5_dense_plan_preserves_full_ten_token_future_at_every_commit():
    tokens = 20
    active_tokens = 5
    rollout_steps = 5
    horizon = torch.tensor([10])
    root = torch.zeros(1, tokens, 4, 5)
    latent = torch.zeros(1, tokens, 3)
    valid = torch.ones(1, tokens, dtype=torch.bool)
    constraint_mask = sample_xz_constraint_mask(
        token_valid_mask=valid,
        initial_active_start=torch.tensor([1]),
        initial_active_end=torch.tensor([6]),
        future_horizon_tokens=horizon,
        rollout_steps=rollout_steps,
        dense_probability=1.0,
        waypoint_probability=0.0,
        goal_probability=0.0,
        max_waypoint_count=4,
    )
    assert constraint_mask[:, 1:20, :, 0].all()

    for commit_offset in range(rollout_steps):
        active_start = 1 + commit_offset
        active_end = active_start + active_tokens
        positions = torch.arange(tokens)[None]
        view = LDFStepView(
            step_index=commit_offset,
            history_end=torch.tensor([active_start]),
            active_start=torch.tensor([active_start]),
            active_end=torch.tensor([active_end]),
            frontier_start=torch.tensor([active_end]),
            timeline_position_ids=positions,
            rope_position_ids=positions - active_start,
            beta=torch.zeros(1, tokens),
        )
        condition = create_xz_condition(
            clean_root_motion=root,
            token_valid_mask=valid,
            constraint_mask=constraint_mask,
            view=view,
            text_context=[torch.zeros(1, 4) for _ in range(tokens)],
            text_null_context=[torch.zeros(1, 4)],
            future_horizon_tokens=horizon,
        )
        history_mask = positions < active_start
        generation_mask = (positions >= active_start) & (positions < active_end)
        inputs = LDFInput(
            noisy_motion=HybridMotion(root, latent),
            beta=torch.zeros(1, tokens),
            history_mask=history_mask,
            generation_mask=generation_mask,
            timeline_position_ids=positions,
            rope_position_ids=positions - active_start,
            previous_root_frame=None,
            previous_root_valid_mask=None,
            condition=condition,
        )

        assert int(condition.future_valid_mask.sum().item()) == 14
        assert int(inputs.future_attention_mask().sum().item()) == 10


def test_one_persistent_future_goal_becomes_active_during_self_forcing():
    root = torch.zeros(1, 10, 4, 5)
    root[..., 3] = 1.0
    valid = torch.ones(1, 10, dtype=torch.bool)
    constraint_mask = torch.zeros_like(root, dtype=torch.bool)
    constraint_mask[:, 7, 2, 0] = True
    constraint_mask[:, 7, 2, 2] = True
    conditions = []

    def builder(view, clean_motion):
        condition = create_xz_condition(
            clean_root_motion=clean_motion.root_motion,
            token_valid_mask=valid,
            constraint_mask=constraint_mask,
            view=view,
            text_context=[torch.zeros(1, 4) for _ in range(10)],
            text_null_context=[torch.zeros(1, 4)],
            future_horizon_tokens=torch.tensor([2]),
        )
        conditions.append(condition)
        return condition

    plan = sample_window_plan(
        {
            "root_motion": root.flatten(1, 2),
            "source_start_token": torch.tensor([4]),
            "span_token_count": torch.tensor([10]),
            "context_token_count": torch.tensor([2]),
            "previous_root_valid_mask": torch.tensor([True]),
        },
        active_tokens=5,
        rollout_steps=2,
        latent_dim=3,
        initial_history_tokens=2,
        phase_offset=torch.zeros(1),
        generator=torch.Generator().manual_seed(13),
    )
    run_training_solver(
        RecordingModel(),
        HybridMotion(root, torch.zeros(1, 10, 3)),
        plan,
        previous_root_frame=torch.zeros(1, 5),
        previous_root_valid_mask=torch.ones(1, dtype=torch.bool),
        condition_builder=builder,
    )

    assert not conditions[0].root_condition_mask.any()
    assert conditions[0].future_timeline_position_ids.tolist() == [[11]]
    assert conditions[0].future_root_condition_mask[0, 0, 2, 0]
    assert conditions[1].root_condition_mask[0, 7, 2, 0]
    # The immutable commit superset still contains this candidate; the dynamic
    # attention mask removes it once token 7 is a visible motion query.
    assert conditions[1].future_timeline_position_ids.tolist() == [[11]]


def test_teacher_replay_is_configurable_without_changing_k_curriculum():
    generator = torch.Generator().manual_seed(0)
    schedule = ((0, 1), (10, 2), (20, 3), (30, 5))
    assert sample_rollout_steps(
        10,
        generator=generator,
        schedule=schedule,
        teacher_replay={2: 1.0},
    ) == 1
    assert sample_rollout_steps(
        10,
        generator=generator,
        schedule=schedule,
        teacher_replay={2: 0.0},
    ) == 2
    assert sample_rollout_steps(
        20,
        generator=generator,
        schedule=schedule,
        teacher_replay={3: 0.0},
    ) == 3
    assert sample_rollout_steps(
        30,
        generator=generator,
        schedule=schedule,
        teacher_replay={5: 0.0},
    ) == 5


def test_teacher_replay_draw_uses_the_generators_device(monkeypatch):
    class DeviceOnlyGenerator:
        device = torch.device("cuda:7")

    captured = {}

    def fake_rand(*shape, device=None, generator=None):
        captured["device"] = device
        captured["generator"] = generator
        return torch.tensor(0.5)

    generator = DeviceOnlyGenerator()
    monkeypatch.setattr("utils.training.ldf.window.torch.rand", fake_rand)
    assert sample_rollout_steps(
        100_000,
        generator=generator,
        schedule=((0, 1), (100_000, 2)),
        teacher_replay={2: 0.0},
    ) == 2
    assert captured == {
        "device": torch.device("cuda:7"),
        "generator": generator,
    }


def test_self_forcing_curriculum_uses_absolute_global_step_boundaries():
    schedule = ((0, 1), (100_000, 2), (200_000, 3), (300_000, 5))
    assert resolve_self_forcing_k(0, schedule) == 1
    assert resolve_self_forcing_k(99_999, schedule) == 1
    assert resolve_self_forcing_k(100_000, schedule) == 2
    assert resolve_self_forcing_k(199_999, schedule) == 2
    assert resolve_self_forcing_k(200_000, schedule) == 3
    assert resolve_self_forcing_k(299_999, schedule) == 3
    assert resolve_self_forcing_k(300_000, schedule) == 5
    assert resolve_self_forcing_k(900_000, schedule) == 5
