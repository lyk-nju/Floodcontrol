import torch

from utils.token_frame import (
    first_future_frame_abs,
    frame_idx_to_token_idx,
    frames_to_token_mask,
    num_frames_for_tokens,
    token_range_to_frame_slice,
    token_start_frame,
)


def test_strict4_mapping_has_no_first_token_exception():
    assert token_start_frame(0) == 0
    assert token_start_frame(1) == 4
    assert num_frames_for_tokens(1) == 4
    assert num_frames_for_tokens(3) == 12
    assert first_future_frame_abs(2) == 8
    assert frame_idx_to_token_idx(0) == 0
    assert frame_idx_to_token_idx(3) == 0
    assert frame_idx_to_token_idx(4) == 1
    assert token_range_to_frame_slice(2, 3) == slice(8, 20)


def test_frame_mask_reduces_non_overlapping_patches():
    mask = torch.tensor([[False, True, False, False, False, False, False, False]])
    assert torch.equal(frames_to_token_mask(mask, 2), torch.tensor([[True, False]]))
