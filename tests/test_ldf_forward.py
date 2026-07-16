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
    prompts = [torch.randn(3, 8) for _ in range(batch)]
    text = [prompt for prompt in prompts for _ in range(tokens)]
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
        previous_root_valid_mask=None,
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


def test_changing_active_xz_condition_changes_root_prediction():
    torch.manual_seed(5)
    model = make_model().eval()
    inputs = make_input(batch=1, tokens=4)
    mask = torch.zeros_like(inputs.noisy_motion.root_motion, dtype=torch.bool)
    mask[..., 0] = True
    mask[..., 2] = True
    first_value = torch.zeros_like(inputs.noisy_motion.root_motion)
    second_value = first_value.clone()
    second_value[..., 0] = 1.5
    second_value[..., 2] = -0.75

    def predict(value):
        condition = LDFCondition(
            inputs.condition.text_context,
            inputs.condition.text_null_context,
            root_condition_value=value,
            root_condition_mask=mask,
        )
        conditioned = LDFInput(**{**inputs.__dict__, "condition": condition})
        return model(conditioned).velocity.root_motion

    assert not torch.allclose(predict(first_value), predict(second_value))


def test_active_xz_condition_preserves_noisy_root_as_an_independent_input():
    torch.manual_seed(6)
    model = make_model().eval()
    inputs = make_input(batch=1, tokens=4)
    mask = torch.zeros_like(inputs.noisy_motion.root_motion, dtype=torch.bool)
    mask[..., 0] = True
    mask[..., 2] = True
    condition = LDFCondition(
        inputs.condition.text_context,
        inputs.condition.text_null_context,
        root_condition_value=torch.zeros_like(inputs.noisy_motion.root_motion),
        root_condition_mask=mask,
    )

    changed_root = inputs.noisy_motion.root_motion.clone()
    changed_root[..., 0] += 17.0
    changed_root[..., 2] -= 17.0
    changed = LDFInput(
        **{
            **inputs.__dict__,
            "noisy_motion": HybridMotion(
                changed_root,
                inputs.noisy_motion.latent_motion,
            ),
            "condition": condition,
        }
    )
    original = LDFInput(**{**inputs.__dict__, "condition": condition})

    with torch.no_grad():
        original_velocity = model(original).velocity.root_motion
        changed_velocity = model(changed).velocity.root_motion
    assert not torch.allclose(original_velocity, changed_velocity)


def test_unobserved_active_root_values_cannot_leak_through_condition_input():
    torch.manual_seed(7)
    model = make_model().eval()
    inputs = make_input(batch=1, tokens=4)
    mask = torch.zeros_like(inputs.noisy_motion.root_motion, dtype=torch.bool)
    mask[..., 0] = True
    mask[..., 2] = True
    first_value = torch.zeros_like(inputs.noisy_motion.root_motion)
    second_value = first_value.clone()
    second_value[..., 1] = 100.0
    second_value[..., 3:] = -100.0

    def predict(value):
        condition = LDFCondition(
            inputs.condition.text_context,
            inputs.condition.text_null_context,
            root_condition_value=value,
            root_condition_mask=mask,
        )
        conditioned = LDFInput(**{**inputs.__dict__, "condition": condition})
        return model(conditioned).velocity.root_motion

    with torch.no_grad():
        assert torch.equal(predict(first_value), predict(second_value))


def test_bfloat16_text_runs_with_float32_model_without_autocast():
    model = make_model().eval()
    inputs = make_input(batch=1, tokens=4)
    condition = LDFCondition(
        [value.bfloat16() for value in inputs.condition.text_context],
        [value.bfloat16() for value in inputs.condition.text_null_context],
    )
    conditioned = LDFInput(**{**inputs.__dict__, "condition": condition})
    with torch.no_grad():
        output = model(conditioned)
    assert output.velocity.root_motion.dtype == torch.float32


def test_body_heading_is_derived_from_clean_root():
    model = make_model()
    inputs = make_input(batch=1, tokens=4)
    clean = torch.zeros_like(inputs.noisy_motion.root_motion)
    clean[..., 3] = 0
    clean[..., 4] = 1
    heading = model._body_heading_condition(clean, inputs)
    assert torch.allclose(heading, torch.tensor([[0.0, 1.0]]))


def test_future_root_uses_generation_centered_rope_without_extending_body():
    model = make_model().eval()
    root = torch.randn(1, 4, 4, 5)
    latent = torch.randn(1, 4, 8)
    text = [torch.randn(3, 8) for _ in range(4)]
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
        previous_root_valid_mask=None,
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


def test_future_root_supports_bfloat16_autocast():
    model = make_model().train()
    root = torch.randn(1, 4, 4, 5)
    latent = torch.randn(1, 4, 8)
    condition = LDFCondition(
        text_context=[torch.randn(3, 8) for _ in range(4)],
        text_null_context=[torch.zeros(1, 8)],
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
        previous_root_valid_mask=None,
        condition=condition,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = model(inputs)

    assert output.velocity.root_motion.shape == (1, 4, 4, 5)
    assert output.velocity.latent_motion.shape == (1, 4, 8)
    assert torch.isfinite(output.velocity.root_motion).all()
    assert torch.isfinite(output.velocity.latent_motion).all()


def test_future_root_is_packed_after_the_visible_motion_prefix():
    model = make_model().eval()
    condition = LDFCondition(
        text_context=[torch.randn(2, 8) for _ in range(4)],
        text_null_context=[torch.zeros(1, 8)],
        future_root_condition_value=torch.zeros(1, 2, 4, 5),
        future_root_condition_mask=torch.ones(1, 2, 4, 5, dtype=torch.bool),
        future_timeline_position_ids=torch.tensor([[9, 10]]),
        future_valid_mask=torch.tensor([[True, True]]),
    )
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.randn(1, 4, 4, 5), torch.randn(1, 4, 8)
        ),
        beta=torch.tensor([[0.0, 0.5, 0.75, 1.0]]),
        history_mask=torch.tensor([[True, False, False, False]]),
        generation_mask=torch.tensor([[False, True, True, False]]),
        timeline_position_ids=torch.tensor([[5, 6, 7, 8]]),
        rope_position_ids=torch.tensor([[-1, 0, 1, 2]]),
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=condition,
    )
    captured = {}

    def capture_root_blocks(self, tokens, **kwargs):
        captured["input_length"] = tokens.shape[1]
        captured["rope_position_ids"] = kwargs["rope_position_ids"].clone()
        captured["text_query_indices"] = kwargs["text_query_indices"].clone()
        return tokens.new_zeros(tokens.shape[0], tokens.shape[1], self.root_patch_dim)

    model.root_transformer._run_blocks = types.MethodType(
        capture_root_blocks, model.root_transformer
    )
    output = model(inputs)

    assert captured["input_length"] == 5
    assert captured["rope_position_ids"].tolist() == [[-1, 0, 1, 3, 4]]
    assert captured["text_query_indices"].tolist() == [[0, 1, 2, -1, -1]]
    assert not output.velocity.root_motion[:, 3].any()


def test_future_root_mask_blocks_unobserved_features_before_projection():
    model = make_model().eval()
    future = torch.arange(20, dtype=torch.float32).reshape(1, 1, 4, 5)
    future_mask = torch.zeros_like(future, dtype=torch.bool)
    future_mask[..., 0] = True
    future_mask[..., 2] = True
    condition = LDFCondition(
        text_context=[torch.zeros(1, 8) for _ in range(2)],
        text_null_context=[torch.zeros(1, 8)],
        future_root_condition_value=future,
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=torch.tensor([[2]]),
        future_valid_mask=torch.tensor([[True]]),
    )
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(1, 2, 4, 5), torch.zeros(1, 2, 8)
        ),
        beta=torch.full((1, 2), 0.5),
        history_mask=torch.zeros(1, 2, dtype=torch.bool),
        generation_mask=torch.ones(1, 2, dtype=torch.bool),
        timeline_position_ids=torch.tensor([[0, 1]]),
        rope_position_ids=torch.tensor([[0, 1]]),
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=condition,
    )
    captured = {}

    def capture_projection(self, value):
        captured["projection_input"] = value.clone()
        return value.new_zeros(*value.shape[:-1], self.out_features)

    model.root_transformer.future_projection.forward = types.MethodType(
        capture_projection, model.root_transformer.future_projection
    )
    model(inputs)

    projected_value = captured["projection_input"][..., :20].reshape(1, 1, 4, 5)
    assert torch.equal(projected_value[..., 0], future[..., 0])
    assert torch.equal(projected_value[..., 2], future[..., 2])
    assert not projected_value[..., 1].any()
    assert not projected_value[..., 3:].any()


def test_changing_future_xz_lookahead_changes_root_prediction():
    torch.manual_seed(17)
    model = make_model().eval()
    inputs = make_input(batch=1, tokens=3)
    future_mask = torch.zeros(1, 2, 4, 5, dtype=torch.bool)
    future_mask[..., 0] = True
    future_mask[..., 2] = True
    first_value = torch.zeros(1, 2, 4, 5)
    second_value = first_value.clone()
    second_value[..., 0] = 2.0
    second_value[..., 2] = -1.0

    def predict(value):
        condition = LDFCondition(
            inputs.condition.text_context,
            inputs.condition.text_null_context,
            future_root_condition_value=value,
            future_root_condition_mask=future_mask,
            future_timeline_position_ids=torch.tensor([[3, 4]]),
            future_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        conditioned = LDFInput(**{**inputs.__dict__, "condition": condition})
        return model(conditioned).velocity.root_motion

    assert not torch.allclose(predict(first_value), predict(second_value))


def test_local_root_uses_per_sample_previous_root_validity():
    model = make_model()
    clean_root = torch.zeros(2, 1, 4, 5)
    clean_root[..., 3] = 1.0
    clean_root[:, 0, :, 0] = torch.arange(4, dtype=torch.float32)
    previous = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    local, valid, _ = model._local_root(
        clean_root,
        previous,
        torch.tensor([False, True]),
    )
    assert torch.equal(valid[0, 0, 0], torch.tensor([False, False, False, True]))
    assert valid[1, 0, 0].all()
    assert torch.allclose(local[0, 0, 0, :3], torch.zeros(3))
    assert torch.allclose(local[1, 0, 0, 1], torch.tensor(20.0))


def test_invisible_motion_and_text_tail_cannot_change_visible_predictions():
    torch.manual_seed(29)
    model = make_model().eval()
    inputs = make_input(batch=1, tokens=6)
    visible = torch.tensor([[True, True, True, True, False, False]])
    inputs = LDFInput(
        **{
            **inputs.__dict__,
            "generation_mask": visible,
        }
    )
    changed_root = inputs.noisy_motion.root_motion.clone()
    changed_latent = inputs.noisy_motion.latent_motion.clone()
    changed_root[:, 4:] += 1000.0
    changed_latent[:, 4:] -= 1000.0
    changed_text = list(inputs.condition.text_context)
    changed_text[4] = torch.full_like(changed_text[4], 1000.0)
    changed_text[5] = torch.full_like(changed_text[5], -1000.0)
    changed_condition = LDFCondition(
        changed_text, inputs.condition.text_null_context
    )
    changed_inputs = LDFInput(
        **{
            **inputs.__dict__,
            "noisy_motion": HybridMotion(changed_root, changed_latent),
            "condition": changed_condition,
        }
    )

    with torch.no_grad():
        original = model(inputs)
        changed = model(changed_inputs)
    assert torch.equal(
        original.velocity.root_motion[:, :4],
        changed.velocity.root_motion[:, :4],
    )
    assert torch.equal(
        original.velocity.latent_motion[:, :4],
        changed.velocity.latent_motion[:, :4],
    )
    assert not original.velocity.root_motion[:, 4:].any()
    assert not original.velocity.latent_motion[:, 4:].any()
