import types

import torch

from models.diffusion_forcing_wan import LDF
from utils.conditions.ldf import HybridMotion, LDFCondition, LDFPrediction
from utils.training.ldf.flow import mix_fixed_noise
from utils.training.ldf.solver import run_persistent_rollout
from utils.training.ldf.window import sample_window_plan


def make_model():
    return LDF(
        latent_dim=3,
        root_mean=[0] * 5,
        root_std=[1] * 5,
        local_root_mean=[0] * 4,
        local_root_std=[1] * 4,
        hidden_dim=8,
        ffn_dim=16,
        freq_dim=8,
        text_dim=4,
        text_len=4,
        num_heads=2,
        root_num_layers=1,
        body_num_layers=1,
        chunk_size=2,
        noise_steps=4,
    )


def constant_prediction(self, inputs, **kwargs):
    root = torch.ones_like(inputs.noisy_motion.root_motion)
    latent = torch.ones_like(inputs.noisy_motion.latent_motion)
    local = torch.zeros(*root.shape[:3], 4, device=root.device)
    valid = torch.ones_like(local, dtype=torch.bool)
    return LDFPrediction(HybridMotion(root, latent), root, local, valid)


def stream_condition(tokens=6):
    prompt = torch.ones(1, 4)
    return LDFCondition([prompt for _ in range(tokens)], [torch.zeros(1, 4)])


def test_ldf_does_not_expose_world_incomplete_multi_token_stream_api():
    model = make_model()
    assert not hasattr(model, "stream_generate")
    assert callable(model.stream_generate_step)


def test_step_input_excludes_the_untouched_pure_noise_frontier():
    model = make_model()
    motion = HybridMotion(torch.zeros(1, 6, 4, 5), torch.zeros(1, 6, 3))
    positions = torch.arange(6)[None]
    condition = stream_condition()
    inputs = model.create_input(
        motion,
        beta=torch.ones(1, 6),
        next_beta=torch.tensor([[0.75, 1.0, 1.0, 1.0, 1.0, 1.0]]),
        timeline_position_ids=positions,
        commit_index=0,
        condition=condition,
        previous_root_frame=None,
        previous_root_valid_mask=None,
    )
    assert inputs.generation_mask.tolist() == [[True, False, False, False, False, False]]
    inputs.validate()


def test_denoise_step_matches_euler_update_and_preserves_ideal_bridge():
    model = make_model()
    clean = HybridMotion(
        torch.randn(1, 6, 4, 5),
        torch.randn(1, 6, 3),
    )
    noise = HybridMotion(
        torch.randn_like(clean.root_motion),
        torch.randn_like(clean.latent_motion),
    )
    beta = torch.tensor([[0.0, 0.4, 0.7, 1.0, 1.0, 1.0]])
    next_beta = torch.tensor([[0.0, 0.2, 0.5, 0.8, 1.0, 1.0]])
    current = mix_fixed_noise(clean, noise, beta)
    target = HybridMotion(
        clean.root_motion - noise.root_motion,
        clean.latent_motion - noise.latent_motion,
    )

    def perfect_forward(self, inputs):
        local = torch.zeros(*inputs.noisy_motion.root_motion.shape[:3], 4)
        return LDFPrediction(
            velocity=target,
            clean_root_motion=clean.root_motion,
            local_root_motion=local,
            local_root_feature_valid=torch.ones_like(local, dtype=torch.bool),
        )

    model.forward = types.MethodType(perfect_forward, model)
    inputs = model.create_input(
        current,
        beta=beta,
        next_beta=next_beta,
        timeline_position_ids=torch.arange(6)[None],
        commit_index=1,
        condition=stream_condition(),
        previous_root_frame=None,
        previous_root_valid_mask=None,
    )
    advanced, prediction = model.denoise_step(
        inputs,
        next_beta,
        use_cfg=False,
    )
    expected = mix_fixed_noise(clean, noise, next_beta)
    # Euler update and direct bridge mixing use different floating-point
    # operation orders; compare at an explicit FP32 numerical tolerance.
    assert torch.allclose(
        advanced.root_motion,
        expected.root_motion,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        advanced.latent_motion,
        expected.latent_motion,
        atol=1e-6,
        rtol=1e-5,
    )
    assert prediction.velocity is target


def test_commit_rebase_preserves_the_world_space_motion_state():
    model = make_model()
    root = torch.randn(2, 6, 4, 5)
    latent = torch.randn(2, 6, 3)
    beta = torch.rand(2, 6)
    token_index = torch.tensor([1, 3])
    motion = HybridMotion(root, latent)

    rebased, committed, translation_xz = model.commit_step(
        motion,
        beta,
        token_index,
    )

    normalized_translation = translation_xz / model.root_std[[0, 2]]
    restored_world_xz = rebased.root_motion[..., [0, 2]] + (
        (1.0 - beta)[..., None, None]
        * normalized_translation[:, None, None, :]
    )

    assert torch.allclose(restored_world_xz, root[..., [0, 2]])
    assert torch.equal(rebased.latent_motion, latent)
    assert torch.allclose(
        committed.root_motion[..., 3:5].norm(dim=-1),
        torch.ones_like(committed.root_motion[..., 3]),
    )
    assert torch.allclose(
        model.denormalize_root(committed.root_motion)[:, 0, -1, [0, 2]],
        translation_xz,
    )


def test_stream_updates_both_fields_and_snapshot_restore_is_deterministic():
    model = make_model()
    model.predict_with_cfg = types.MethodType(constant_prediction, model)
    generator = torch.Generator().manual_seed(7)
    initial = HybridMotion(torch.zeros(1, 6, 4, 5), torch.zeros(1, 6, 3))
    condition = stream_condition()
    state = model.init_stream_state(
        batch_size=1,
        window_tokens=6,
        generator=generator,
        initial_noise=initial,
    )
    state, committed = model.stream_generate_step(state, condition)
    assert state.commit_index == 1
    assert torch.allclose(
        committed.root_motion[..., :3],
        torch.ones_like(committed.root_motion[..., :3]),
    )
    assert torch.allclose(
        committed.root_motion[..., 3:5].norm(dim=-1),
        torch.ones_like(committed.root_motion[..., 3]),
    )
    assert torch.allclose(committed.latent_motion, torch.ones_like(committed.latent_motion))

    snapshot = model.create_stream_snapshot(state)
    continued, committed_a = model.stream_generate_step(state, condition)
    restored = model.create_stream_state_from_snapshot(snapshot)
    replayed, committed_b = model.stream_generate_step(restored, condition)
    assert torch.equal(committed_a.root_motion, committed_b.root_motion)
    assert torch.equal(committed_a.latent_motion, committed_b.latent_motion)
    assert torch.equal(continued.noisy_motion.root_motion, replayed.noisy_motion.root_motion)


def test_real_ldf_persistent_cold_matches_two_runtime_commits():
    torch.manual_seed(19)
    model = make_model().eval()
    tokens = 6
    clean = HybridMotion(
        torch.randn(1, tokens, 4, 5),
        torch.randn(1, tokens, model.latent_dim),
    )
    clean.root_motion[..., 3] = 1.0
    clean.root_motion[..., 4] = 0.0
    plan = sample_window_plan(
        {
            "root_motion": clean.root_motion.flatten(1, 2),
            "source_start_token": torch.zeros(1, dtype=torch.long),
            "span_token_count": torch.full((1,), tokens, dtype=torch.long),
            "context_token_count": torch.zeros(1, dtype=torch.long),
            "previous_root_valid_mask": torch.zeros(1, dtype=torch.bool),
        },
        active_tokens=model.chunk_size,
        rollout_steps=2,
        latent_dim=model.latent_dim,
        initial_history_tokens=0,
        phase_offset=torch.zeros(1),
        allow_cold_start=True,
        generator=torch.Generator().manual_seed(23),
    )
    condition = stream_condition(tokens)

    with torch.no_grad():
        _, _, training_first_state, training_first_commits = run_persistent_rollout(
            model,
            clean,
            plan,
            previous_root_frame=None,
            previous_root_valid_mask=None,
            condition_builder=lambda _view, _clean: condition,
            supervised_microstep=model.noise_steps - 1,
        )
        _, _, training_state, training_commits = run_persistent_rollout(
            model,
            clean,
            plan,
            previous_root_frame=None,
            previous_root_valid_mask=None,
            condition_builder=lambda _view, _clean: condition,
            supervised_microstep=5,
        )
        runtime_state = model.init_stream_state(
            batch_size=1,
            window_tokens=tokens,
            initial_noise=plan.noise,
            generator=torch.Generator().manual_seed(29),
        )
        runtime_commits = []
        runtime_state, committed = model.stream_generate_step(
            runtime_state,
            condition,
            roll_window=False,
            cfg_mode="nocfg",
        )
        runtime_commits.append(committed)
        runtime_first_state = runtime_state
        runtime_state, committed = model.stream_generate_step(
            runtime_state,
            condition,
            roll_window=False,
            cfg_mode="nocfg",
        )
        runtime_commits.append(committed)

    assert len(training_first_commits) == 1
    assert torch.allclose(
        training_first_commits[0].root_motion,
        runtime_commits[0].root_motion,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        training_first_state.noisy_motion.root_motion,
        runtime_first_state.noisy_motion.root_motion,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        training_first_state.noisy_motion.latent_motion,
        runtime_first_state.noisy_motion.latent_motion,
        atol=1e-6,
        rtol=1e-6,
    )
    assert training_first_state.current_denoise_step.tolist() == [
        runtime_first_state.current_step
    ]
    assert len(training_commits) == len(runtime_commits) == 2
    for trained, runtime in zip(training_commits, runtime_commits):
        assert torch.allclose(
            trained.root_motion, runtime.root_motion, atol=1e-6, rtol=1e-6
        )
        assert torch.allclose(
            trained.latent_motion, runtime.latent_motion, atol=1e-6, rtol=1e-6
        )
    assert torch.allclose(
        training_state.noisy_motion.root_motion,
        runtime_state.noisy_motion.root_motion,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        training_state.noisy_motion.latent_motion,
        runtime_state.noisy_motion.latent_motion,
        atol=1e-6,
        rtol=1e-6,
    )
    assert training_state.current_denoise_step.tolist() == [runtime_state.current_step]
    runtime_beta = model.triangular_beta(
        timeline_position_ids=torch.arange(tokens)[None],
        diffusion_time=runtime_state.current_step / float(model.noise_steps),
    )
    assert torch.equal(training_state.beta, runtime_beta)
    runtime_origin = sum(
        (
            model.denormalize_root(committed.root_motion)[:, 0, -1, [0, 2]]
            for committed in runtime_commits
        ),
        torch.zeros(1, 2),
    )
    assert torch.allclose(
        training_state.origin_xz, runtime_origin, atol=1e-6, rtol=1e-6
    )


def test_stream_roll_keeps_boundary_and_advances_origin():
    model = make_model()
    model.predict_with_cfg = types.MethodType(constant_prediction, model)
    condition = stream_condition()
    state = model.init_stream_state(
        batch_size=1,
        window_tokens=6,
        generator=torch.Generator().manual_seed(3),
    )
    for _ in range(5):
        state, _ = model.stream_generate_step(state, condition)
    assert state.window_origin == 2
    assert state.epoch == 1
    assert state.previous_root_frame.shape == (1, 5)
