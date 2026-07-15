import pytest
import torch

from utils.token_frame import (
    FRAMES_PER_TOKEN,
    aligned_frame_floor,
    commit_index_to_frame_count,
    frame_count_to_token_count,
    frame_index_to_token_index,
    frame_observation_to_token_mask,
    frame_valid_to_token_valid,
    prefix_valid_token_count,
    previous_committed_frame_index,
    require_aligned_frame_count,
    token_count_to_frame_count,
    token_index_to_frame_slice,
    token_index_to_frame_start,
    token_range_to_frame_slice,
)


def test_four_frame_mapping_has_no_first_token_exception():
    assert FRAMES_PER_TOKEN == 4
    assert token_index_to_frame_start(0) == 0
    assert token_index_to_frame_start(1) == 4
    assert token_index_to_frame_slice(2) == slice(8, 12)
    assert token_count_to_frame_count(3) == 12
    assert frame_count_to_token_count(12) == 3
    assert frame_index_to_token_index(0) == 0
    assert frame_index_to_token_index(3) == 0
    assert frame_index_to_token_index(4) == 1
    assert token_range_to_frame_slice(2, 3) == slice(8, 20)


def test_exact_conversions_reject_negative_and_unaligned_values():
    with pytest.raises(ValueError, match="non-negative"):
        token_index_to_frame_start(-1)
    with pytest.raises(ValueError, match="divisible"):
        require_aligned_frame_count(5)
    with pytest.raises(ValueError, match="divisible"):
        frame_count_to_token_count(5)
    with pytest.raises(TypeError, match="integer"):
        token_count_to_frame_count(1.5)
    assert aligned_frame_floor(7) == 4


def test_commit_boundary_represents_cold_start_without_fake_frame_zero():
    assert commit_index_to_frame_count(0) == 0
    assert previous_committed_frame_index(0) is None
    assert commit_index_to_frame_count(2) == 8
    assert previous_committed_frame_index(2) == 7


def test_motion_validity_and_sparse_observations_use_different_reductions():
    mask = torch.tensor([[True, True, True, False, False, True, False, False]])
    assert torch.equal(
        frame_valid_to_token_valid(mask),
        torch.tensor([[False, False]]),
    )
    assert torch.equal(
        frame_observation_to_token_mask(mask),
        torch.tensor([[True, True]]),
    )

    feature_mask = mask[:, :, None].expand(-1, -1, 3)
    reduced = frame_observation_to_token_mask(feature_mask, frame_dim=1)
    assert reduced.shape == (1, 2, 3)


def test_frame_masks_must_be_boolean_and_patch_aligned():
    with pytest.raises(TypeError, match="dtype bool"):
        frame_valid_to_token_valid(torch.ones(1, 4))
    with pytest.raises(ValueError, match="divisible"):
        frame_valid_to_token_valid(torch.ones(1, 5, dtype=torch.bool))


def test_prefix_count_rejects_holes_in_validity():
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.equal(prefix_valid_token_count(mask), torch.tensor([2, 1]))
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        prefix_valid_token_count(torch.tensor([[True, False, True]]))
