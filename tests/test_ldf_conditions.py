import math

import pytest
import torch

from utils.conditions.ldf import (
    HybridMotion,
    LDFCondition,
    LDFInput,
    create_cfg_condition,
    create_ldf_condition,
    create_window_condition,
)
from utils.motion_process import recover_local_root


def _text(batch=1, dim=8):
    return [torch.ones(2, dim) for _ in range(batch)], [torch.zeros(1, dim) for _ in range(batch)]


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
    text, null = _text()
    value = torch.zeros(1, 2, 4, 5)
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[..., 3] = True
    condition = LDFCondition(text, null, value, mask)
    with pytest.raises(ValueError, match="cos/sin"):
        condition.validate(batch_size=1, token_length=2, latent_dim=8)


def test_window_compile_packs_sparse_future_tokens_and_cfg_is_read_only():
    text, null = _text()
    root = torch.zeros(1, 24, 5)
    mask = torch.zeros_like(root, dtype=torch.bool)
    mask[:, 20:24, :3] = True
    window = create_window_condition(
        text_context=text,
        text_null_context=null,
        window_origin=0,
        window_tokens=4,
        future_tokens=2,
        root_condition_value=root,
        root_condition_mask=mask,
    )
    condition = create_ldf_condition(window)
    assert condition.root_condition_value.shape == (1, 4, 4, 5)
    assert condition.future_valid_mask.tolist() == [[True, False]]
    original = condition.future_valid_mask.clone()
    branches = create_cfg_condition(condition)
    assert torch.equal(condition.future_valid_mask, original)
    assert not branches["history"].future_valid_mask.any()
    assert branches["constraint"].future_valid_mask.any()


def test_timeline_and_rope_positions_remain_distinct_after_window_roll():
    text, null = _text()
    root = torch.zeros(1, 40, 5)
    mask = torch.zeros_like(root, dtype=torch.bool)
    # window_origin=3 and window_tokens=4 place future tokens at absolute 7 and 8.
    mask[:, 28:32, :3] = True
    window = create_window_condition(
        text_context=text,
        text_null_context=null,
        window_origin=3,
        window_tokens=4,
        future_tokens=2,
        root_condition_value=root,
        root_condition_mask=mask,
    )
    condition = create_ldf_condition(window)
    assert condition.future_timeline_position_ids.tolist() == [[7, 0]]

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
    ).tolist() == [[3, -4]]


def test_future_timeline_positions_cannot_overlap_current_window():
    text, null = _text()
    condition = LDFCondition(
        text_context=text,
        text_null_context=null,
        future_root_condition_value=torch.zeros(1, 1, 4, 5),
        future_root_condition_mask=torch.ones(1, 1, 4, 5, dtype=torch.bool),
        future_timeline_position_ids=torch.tensor([[3]]),
        future_valid_mask=torch.tensor([[True]]),
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
    with pytest.raises(ValueError, match="after the current motion window"):
        inputs.validate()


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
