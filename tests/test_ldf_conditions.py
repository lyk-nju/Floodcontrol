import math

import pytest
import torch

from utils.conditions.ldf import (
    HybridMotion,
    LDFCondition,
    LDFInput,
    create_cfg_condition,
    create_ldf_condition,
    expand_null_timeline,
)
from utils.motion_process import recover_local_root


def _text(batch=1, tokens=1, dim=8):
    prompts = [torch.ones(2, dim) for _ in range(batch)]
    timeline = [prompt for prompt in prompts for _ in range(tokens)]
    return timeline, [torch.zeros(1, dim) for _ in range(batch)]


def test_backward_local_root_cold_start_and_forward_motion():
    root = torch.zeros(1, 1, 4, 5)
    root[..., 3] = 1.0
    root[0, 0, :, 2] = torch.arange(4) * 0.1
    values, valid = recover_local_root(root.flatten(1, 2), None, fps=20.0)
    assert torch.equal(valid[0, 0, 0], torch.tensor([False, False, False, True]))
    assert torch.allclose(values[0, 0, 0, :3], torch.zeros(3))
    assert torch.allclose(values[0, 0, 1:, 2], torch.full((3,), 2.0))


def test_local_velocity_is_invariant_to_global_yaw_rotation():
    root = torch.zeros(1, 1, 4, 5)
    root[..., 3] = 1.0
    root[0, 0, :, 2] = torch.arange(4) * 0.1
    original, _ = recover_local_root(root.flatten(1, 2), None)

    angle = math.pi / 2
    rotated = root.clone()
    x, z = root[..., 0], root[..., 2]
    rotated[..., 0] = math.cos(angle) * x + math.sin(angle) * z
    rotated[..., 2] = -math.sin(angle) * x + math.cos(angle) * z
    rotated[..., 3] = math.cos(angle)
    rotated[..., 4] = math.sin(angle)
    transformed, _ = recover_local_root(rotated.flatten(1, 2), None)
    assert torch.allclose(original[..., :3], transformed[..., :3], atol=1e-5)


def test_heading_mask_must_be_paired():
    text, null = _text(tokens=2)
    value = torch.zeros(1, 2, 4, 5)
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[..., 3] = True
    condition = LDFCondition(text, null, value, mask)
    with pytest.raises(ValueError, match="cos/sin"):
        condition.validate(batch_size=1, token_length=2, latent_dim=8)


def test_window_compile_packs_sparse_future_tokens_and_cfg_is_read_only():
    text, null = _text(tokens=4)
    active_root = torch.zeros(1, 16, 5)
    active_mask = torch.zeros_like(active_root, dtype=torch.bool)
    future_root = torch.zeros(1, 8, 5)
    future_mask = torch.zeros_like(future_root, dtype=torch.bool)
    future_mask[:, :4, :3] = True
    condition = create_ldf_condition(
        text_context=text,
        text_null_context=null,
        root_condition_value=active_root,
        root_condition_mask=active_mask,
        future_root_condition_value=future_root,
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=torch.tensor([4, 5]),
        future_horizon_tokens=2,
    )
    assert condition.root_condition_value.shape == (1, 4, 4, 5)
    assert condition.future_valid_mask.tolist() == [[True]]
    original = condition.future_valid_mask.clone()
    branches = create_cfg_condition(condition, token_length=4)
    assert torch.equal(condition.future_valid_mask, original)
    assert not branches["history"].future_valid_mask.any()
    assert branches["constraint"].future_valid_mask.any()


def test_timeline_and_rope_positions_remain_distinct_after_window_roll():
    text, null = _text(tokens=4)
    root = torch.zeros(1, 16, 5)
    mask = torch.zeros_like(root, dtype=torch.bool)
    future_root = torch.zeros(1, 8, 5)
    future_mask = torch.zeros_like(future_root, dtype=torch.bool)
    # window_origin=3 and window_tokens=4 place future tokens at absolute 7 and 8.
    future_mask[:, :4, :3] = True
    condition = create_ldf_condition(
        text_context=text,
        text_null_context=null,
        root_condition_value=root,
        root_condition_mask=mask,
        future_root_condition_value=future_root,
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=torch.tensor([7, 8]),
        future_horizon_tokens=2,
    )
    assert condition.future_timeline_position_ids.tolist() == [[7]]

    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(1, 4, 4, 5), torch.zeros(1, 4, 8)
        ),
        beta=torch.tensor([[0.0, 0.5, 0.75, 1.0]]),
        history_mask=torch.tensor([[True, False, False, False]]),
        generation_mask=torch.tensor([[False, True, True, True]]),
        timeline_position_ids=torch.tensor([[3, 4, 5, 6]]),
        rope_position_ids=torch.tensor([[-1, 0, 1, 2]]),
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=condition,
    )
    inputs.validate()
    assert inputs.rope_origin.tolist() == [[4]]
    assert inputs.timeline_to_rope(
        condition.future_timeline_position_ids
    ).tolist() == [[3]]


def test_future_candidate_overlap_is_removed_from_attention_view():
    text, null = _text(tokens=4)
    condition = LDFCondition(
        text_context=text,
        text_null_context=null,
        future_root_condition_value=torch.zeros(1, 1, 4, 5),
        future_root_condition_mask=torch.ones(1, 1, 4, 5, dtype=torch.bool),
        future_timeline_position_ids=torch.tensor([[3]]),
        future_valid_mask=torch.tensor([[True]]),
        future_horizon_tokens=torch.tensor([1]),
    )
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(1, 4, 4, 5), torch.zeros(1, 4, 8)
        ),
        beta=torch.ones(1, 4),
        history_mask=torch.zeros(1, 4, dtype=torch.bool),
        generation_mask=torch.ones(1, 4, dtype=torch.bool),
        timeline_position_ids=torch.arange(4)[None],
        rope_position_ids=torch.arange(4)[None],
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=condition,
    )
    inputs.validate()
    assert inputs.future_attention_mask().tolist() == [[False]]


def test_dynamic_future_selection_is_per_sample_and_preserves_candidate_superset():
    prompt = torch.ones(1, 8)
    null = torch.zeros(1, 8)
    candidate_positions = torch.tensor(
        [list(range(1, 8)), list(range(11, 18))], dtype=torch.long
    )
    candidate_valid = torch.ones(2, 7, dtype=torch.bool)
    condition = LDFCondition(
        text_context=[prompt for _ in range(16)],
        text_null_context=[null, null],
        future_root_condition_value=torch.zeros(2, 7, 4, 5),
        future_root_condition_mask=torch.ones(2, 7, 4, 5, dtype=torch.bool),
        future_timeline_position_ids=candidate_positions,
        future_valid_mask=candidate_valid,
        future_horizon_tokens=torch.tensor([3, 3]),
    )
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(2, 8, 4, 5), torch.zeros(2, 8, 8)
        ),
        beta=torch.ones(2, 8),
        history_mask=torch.zeros(2, 8, dtype=torch.bool),
        generation_mask=torch.tensor(
            [
                [True, False, False, False, False, False, False, False],
                [True, True, True, True, False, False, False, False],
            ]
        ),
        timeline_position_ids=torch.stack((torch.arange(8), torch.arange(10, 18))),
        rope_position_ids=torch.arange(8)[None].expand(2, -1),
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=condition,
    )

    selected = inputs.future_attention_mask()
    assert candidate_positions[0, selected[0]].tolist() == [1, 2, 3]
    assert candidate_positions[1, selected[1]].tolist() == [14, 15, 16]
    assert torch.equal(condition.future_valid_mask, candidate_valid)


def test_future_valid_mask_must_match_observed_constraint_tokens():
    text, null = _text(tokens=4)
    condition = LDFCondition(
        text_context=text,
        text_null_context=null,
        future_root_condition_value=torch.zeros(1, 2, 4, 5),
        future_root_condition_mask=torch.zeros(1, 2, 4, 5, dtype=torch.bool),
        future_timeline_position_ids=torch.tensor([[4, 5]]),
        future_valid_mask=torch.tensor([[True, False]]),
        future_horizon_tokens=torch.tensor([2]),
    )
    with pytest.raises(ValueError, match="exactly match constrained future tokens"):
        condition.validate(batch_size=1, token_length=4, latent_dim=8)


def test_previous_root_frame_and_validity_mask_are_paired():
    text, null = _text()
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(1, 1, 4, 5), torch.zeros(1, 1, 8)
        ),
        beta=torch.ones(1, 1),
        history_mask=torch.zeros(1, 1, dtype=torch.bool),
        generation_mask=torch.ones(1, 1, dtype=torch.bool),
        timeline_position_ids=torch.zeros(1, 1, dtype=torch.long),
        rope_position_ids=torch.zeros(1, 1, dtype=torch.long),
        previous_root_frame=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]]),
        previous_root_valid_mask=None,
        condition=LDFCondition(text, null),
    )
    with pytest.raises(ValueError, match="must both be set"):
        inputs.validate()


def test_conditional_text_is_strictly_token_aligned_and_cfg_expands_nulls():
    prompt = torch.ones(2, 8)
    null = torch.zeros(1, 8)
    static = LDFCondition([prompt], [null])
    with pytest.raises(ValueError, match=r"B\*T=3"):
        static.validate_structure(batch_size=1, token_length=3, latent_dim=8)

    timeline = [prompt, prompt, prompt]
    condition = LDFCondition(timeline, [null])
    condition.validate_structure(batch_size=1, token_length=3, latent_dim=8)
    assert all(item is prompt for item in condition.text_context)
    expanded = expand_null_timeline(condition.text_null_context, 3)
    assert all(item is null for item in expanded)

    branches = create_cfg_condition(condition, token_length=3)
    assert branches["joint"] is condition
    assert all(item is null for item in branches["history"].text_context)
    assert all(item is null for item in branches["constraint"].text_context)
    assert all(item is prompt for item in branches["text"].text_context)


def test_structure_validation_does_not_replace_full_semantic_validation():
    text, null = _text(tokens=2)
    inputs = LDFInput(
        noisy_motion=HybridMotion(
            torch.zeros(1, 2, 4, 5), torch.zeros(1, 2, 8)
        ),
        beta=torch.tensor([[1.5, 1.5]]),
        history_mask=torch.zeros(1, 2, dtype=torch.bool),
        generation_mask=torch.ones(1, 2, dtype=torch.bool),
        timeline_position_ids=torch.arange(2)[None],
        rope_position_ids=torch.arange(2)[None],
        previous_root_frame=None,
        previous_root_valid_mask=None,
        condition=LDFCondition(text, null),
    )
    inputs.validate_structure()
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        inputs.validate()
