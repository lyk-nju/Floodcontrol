import types

import torch

from models.diffusion_forcing_wan import LDF
from utils.conditions.ldf import HybridMotion, LDFCondition, LDFPrediction


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


def test_step_input_excludes_the_untouched_pure_noise_frontier():
    model = make_model()
    motion = HybridMotion(torch.zeros(1, 6, 4, 5), torch.zeros(1, 6, 3))
    positions = torch.arange(6)[None]
    condition = LDFCondition([torch.ones(1, 4)], [torch.zeros(1, 4)])
    inputs = model._create_step_input(
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


def test_stream_updates_both_fields_and_snapshot_restore_is_deterministic():
    model = make_model()
    model.predict_with_cfg = types.MethodType(constant_prediction, model)
    generator = torch.Generator().manual_seed(7)
    initial = HybridMotion(torch.zeros(1, 6, 4, 5), torch.zeros(1, 6, 3))
    condition = LDFCondition([torch.ones(1, 4)], [torch.zeros(1, 4)])
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


def test_stream_roll_keeps_boundary_and_advances_origin():
    model = make_model()
    model.predict_with_cfg = types.MethodType(constant_prediction, model)
    condition = LDFCondition([torch.ones(1, 4)], [torch.zeros(1, 4)])
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
