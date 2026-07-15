import torch

from models.tools.wan_model import WanCrossAttention


def test_token_aligned_cross_attention_isolates_prompt_contexts():
    torch.manual_seed(4)
    attention = WanCrossAttention(8, 2).eval()
    value = torch.randn(1, 2, 8)
    context = torch.randn(1, 2, 3, 8)
    lengths = torch.tensor([[3, 3]])
    query_lengths = torch.tensor([2])
    query_mask = torch.ones(1, 2, dtype=torch.bool)

    original = attention(
        value, context, lengths, query_lengths, query_mask=query_mask
    )
    changed_context = context.clone()
    changed_context[:, 1] += 10.0
    changed = attention(
        value, changed_context, lengths, query_lengths, query_mask=query_mask
    )

    assert torch.equal(original[:, 0], changed[:, 0])
    assert not torch.equal(original[:, 1], changed[:, 1])


def test_token_aligned_cross_attention_masks_non_motion_queries():
    attention = WanCrossAttention(8, 2).eval()
    value = torch.randn(1, 3, 8)
    context = torch.randn(1, 3, 2, 8)
    output = attention(
        value,
        context,
        torch.tensor([[2, 2, 0]]),
        torch.tensor([3]),
        query_mask=torch.tensor([[True, True, False]]),
    )
    assert not output[:, :2].eq(0).all()
    assert not output[:, 2].any()
