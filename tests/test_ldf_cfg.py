import types

import torch

from models.diffusion_forcing_wan import LDF
from utils.conditions.ldf import HybridMotion, LDFCondition, LDFInput


def test_separated_cfg_combination_and_shared_root_boundary():
    model = LDF(
        latent_dim=2,
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
        chunk_size=1,
        noise_steps=1,
    )
    text = [torch.ones(1, 4) for _ in range(2)]
    null = [torch.zeros(1, 4)]
    root_value = torch.zeros(1, 2, 4, 5)
    root_mask = torch.ones_like(root_value, dtype=torch.bool)
    condition = LDFCondition(text, null, root_value, root_mask)
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(1, 2, 4, 5), torch.zeros(1, 2, 2)
        ),
        beta=torch.ones(1, 2),
        history_mask=torch.zeros(1, 2, dtype=torch.bool),
        generation_mask=torch.ones(1, 2, dtype=torch.bool),
        timeline_position_ids=torch.arange(2)[None],
        rope_position_ids=torch.arange(2)[None],
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=condition,
    )
    seen_local = []

    def branch_value(cond):
        has_text = bool(cond.text_context[0].abs().sum())
        has_constraint = cond.root_condition_mask is not None and bool(cond.root_condition_mask.any())
        return 1.0 * has_text + 2.0 * has_constraint

    def root_branch(self, call_inputs, cond):
        return torch.full_like(call_inputs.noisy_motion.root_motion, branch_value(cond))

    def body_branch(self, call_inputs, cond, local, valid):
        seen_local.append(local.clone())
        return torch.full_like(call_inputs.noisy_motion.latent_motion, branch_value(cond))

    model._predict_root = types.MethodType(root_branch, model)
    model._predict_body = types.MethodType(body_branch, model)
    output = model.predict_with_cfg(
        inputs, mode="separated", cfg_scale_text=2.0, cfg_scale_constraint=3.0
    )
    assert torch.allclose(output.raw_root_output, torch.full_like(output.raw_root_output, 8.0))
    assert torch.allclose(output.raw_body_output, torch.full_like(output.raw_body_output, 8.0))
    assert torch.allclose(
        output.clean_motion.root_motion[..., 3:5].norm(dim=-1),
        torch.ones_like(output.clean_motion.root_motion[..., 3]),
    )
    assert len(seen_local) == 3
    assert all(torch.equal(seen_local[0], item) for item in seen_local[1:])


def test_cfg_formula_scale_zero_returns_history():
    history = torch.randn(2, 3)
    text = torch.randn(2, 3)
    constraint = torch.randn(2, 3)
    result = LDF._compose_cfg(
        history, text, constraint, scale_text=0.0, scale_constraint=0.0
    )
    assert torch.equal(result, history)
