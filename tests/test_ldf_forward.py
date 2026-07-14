import types

import torch

from models.diffusion_forcing_wan import LDF
from utils.conditions.ldf import HybridMotion, LDFCondition, LDFInput


def make_model():
    return LDF(
        latent_dim=8,
        root_mean=[0] * 5,
        root_std=[1] * 5,
        local_root_mean=[0] * 4,
        local_root_std=[1] * 4,
        hidden_dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=8,
        text_len=8,
        num_heads=4,
        root_num_layers=1,
        body_num_layers=1,
        chunk_size=2,
        noise_steps=4,
    )


def make_input(batch=2, tokens=6):
    root = torch.randn(batch, tokens, 4, 5)
    latent = torch.randn(batch, tokens, 8)
    text = [torch.randn(3, 8) for _ in range(batch)]
    null = [torch.zeros(1, 8) for _ in range(batch)]
    condition = LDFCondition(text, null)
    return LDFInput(
        noisy_motion=HybridMotion(root, latent),
        beta=torch.full((batch, tokens), 0.5),
        history_mask=torch.zeros(batch, tokens, dtype=torch.bool),
        generation_mask=torch.ones(batch, tokens, dtype=torch.bool),
        timeline_position_ids=torch.arange(tokens)[None].expand(batch, -1),
        rope_position_ids=torch.arange(tokens)[None].expand(batch, -1),
        previous_root_frame=None,
        condition=condition,
    )


def test_forward_shapes_and_v_to_x0_identity():
    model = make_model()
    inputs = make_input()
    output = model(inputs)
    assert output.velocity.root_motion.shape == inputs.noisy_motion.root_motion.shape
    assert output.velocity.latent_motion.shape == inputs.noisy_motion.latent_motion.shape
    expected = inputs.noisy_motion.root_motion + inputs.beta[..., None, None] * output.velocity.root_motion
    # x/y/z are unaffected by the heading manifold projection under identity stats.
    assert torch.allclose(output.clean_root_motion[..., :3], expected[..., :3], atol=1e-5)
    assert output.local_root_motion.shape == (2, 6, 4, 4)


def test_body_loss_does_not_backpropagate_into_root_transformer():
    model = make_model().train()
    output = model(make_input(batch=1, tokens=4))
    output.velocity.latent_motion.square().mean().backward()
    assert all(parameter.grad is None for parameter in model.root_transformer.parameters())
    assert any(parameter.grad is not None for parameter in model.body_transformer.parameters())


def test_constraint_view_does_not_mutate_noisy_state():
    model = make_model()
    inputs = make_input(batch=1, tokens=4)
    root_before = inputs.noisy_motion.root_motion.clone()
    value = torch.zeros_like(root_before)
    mask = torch.zeros_like(root_before, dtype=torch.bool)
    mask[:, 1, :, :3] = True
    condition = LDFCondition(
        inputs.condition.text_context,
        inputs.condition.text_null_context,
        root_condition_value=value,
        root_condition_mask=mask,
    )
    model(LDFInput(**{**inputs.__dict__, "condition": condition}))
    assert torch.equal(inputs.noisy_motion.root_motion, root_before)


def test_future_root_uses_generation_centered_rope_without_extending_body():
    model = make_model().eval()
    root = torch.randn(1, 4, 4, 5)
    latent = torch.randn(1, 4, 8)
    text = [torch.randn(3, 8)]
    null = [torch.zeros(1, 8)]
    condition = LDFCondition(
        text_context=text,
        text_null_context=null,
        future_root_condition_value=torch.zeros(1, 2, 4, 5),
        future_root_condition_mask=torch.ones(1, 2, 4, 5, dtype=torch.bool),
        future_timeline_position_ids=torch.tensor([[9, 10]]),
        future_valid_mask=torch.tensor([[True, True]]),
    )
    inputs = LDFInput(
        noisy_motion=HybridMotion(root, latent),
        beta=torch.tensor([[0.0, 0.5, 0.75, 1.0]]),
        history_mask=torch.tensor([[True, False, False, False]]),
        generation_mask=torch.tensor([[False, True, True, True]]),
        timeline_position_ids=torch.tensor([[5, 6, 7, 8]]),
        rope_position_ids=torch.tensor([[-1, 0, 1, 2]]),
        previous_root_frame=None,
        condition=condition,
    )
    captured = {}

    def capture_root_blocks(self, tokens, **kwargs):
        captured["input_length"] = tokens.shape[1]
        captured["rope_position_ids"] = kwargs["rope_position_ids"].clone()
        return tokens.new_zeros(tokens.shape[0], tokens.shape[1], self.root_patch_dim)

    model.root_transformer._run_blocks = types.MethodType(
        capture_root_blocks, model.root_transformer
    )
    output = model(inputs)
    assert captured["input_length"] == 6
    assert captured["rope_position_ids"].tolist() == [[-1, 0, 1, 2, 3, 4]]
    assert output.velocity.root_motion.shape[1] == 4
    assert output.velocity.latent_motion.shape[1] == 4
