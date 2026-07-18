import torch
import pytest
from torch.testing import assert_close

import models.tools.wan_model as wan_model_tools

from models.tools.attention import (
    FLASH_ATTN_2_AVAILABLE,
    _pack_valid_prefix,
    flash_attention,
)
from models.tools.wan_model import WanCrossAttention, embed_text_context


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


def _dense_prompt_context(
    context_bank: torch.Tensor,
    context_lens: torch.Tensor,
    prompt_ids: torch.Tensor,
    query_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_ids = prompt_ids.clamp_min(0)
    batch, length = prompt_ids.shape
    dense_context = context_bank.index_select(0, safe_ids.reshape(-1)).reshape(
        batch,
        length,
        context_bank.shape[1],
        context_bank.shape[2],
    )
    dense_lens = context_lens.index_select(0, safe_ids.reshape(-1)).reshape(
        batch, length
    )
    dense_lens = torch.where(query_mask, dense_lens, torch.zeros_like(dense_lens))
    return dense_context, dense_lens


def test_prompt_bank_cross_attention_matches_dense_token_context():
    torch.manual_seed(23)
    dense_attention = WanCrossAttention(8, 2).eval()
    grouped_attention = WanCrossAttention(8, 2).eval()
    grouped_attention.load_state_dict(dense_attention.state_dict())

    dense_value = torch.randn(2, 4, 8, requires_grad=True)
    grouped_value = dense_value.detach().clone().requires_grad_(True)
    dense_bank = torch.randn(3, 5, 8, requires_grad=True)
    grouped_bank = dense_bank.detach().clone().requires_grad_(True)
    context_lens = torch.tensor([5, 3, 4])
    prompt_ids = torch.tensor([[0, 0, 1, -1], [2, 1, -1, -1]])
    query_lens = torch.tensor([4, 3])
    query_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]]
    )
    dense_context, dense_lens = _dense_prompt_context(
        dense_bank,
        context_lens,
        prompt_ids,
        query_mask,
    )

    dense_output = dense_attention(
        dense_value,
        dense_context,
        dense_lens,
        query_lens,
        query_mask=query_mask,
    )
    grouped_output = grouped_attention(
        grouped_value,
        grouped_bank,
        context_lens,
        query_lens,
        query_mask=query_mask,
        prompt_ids=prompt_ids,
    )
    assert_close(grouped_output, dense_output, atol=1e-6, rtol=1e-5)
    assert not grouped_output[~query_mask].any()

    dense_output.square().sum().backward()
    grouped_output.square().sum().backward()
    assert_close(grouped_value.grad, dense_value.grad, atol=1e-6, rtol=1e-5)
    assert_close(grouped_bank.grad, dense_bank.grad, atol=1e-6, rtol=1e-5)
    for dense_parameter, grouped_parameter in zip(
        dense_attention.parameters(), grouped_attention.parameters()
    ):
        assert_close(
            grouped_parameter.grad,
            dense_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        )


def test_prompt_bank_projects_key_value_once_per_used_prompt():
    torch.manual_seed(29)
    attention = WanCrossAttention(8, 2).eval()
    value = torch.randn(2, 5, 8)
    context_bank = torch.randn(3, 4, 8)
    prompt_ids = torch.tensor([[0, 0, 0, 1, 1], [2, 2, 0, 1, 2]])
    seen_key_shapes: list[tuple[int, ...]] = []
    handle = attention.k.register_forward_pre_hook(
        lambda _module, arguments: seen_key_shapes.append(tuple(arguments[0].shape))
    )
    try:
        output = attention(
            value,
            context_bank,
            torch.tensor([4, 3, 2]),
            torch.tensor([5, 5]),
            prompt_ids=prompt_ids,
        )
    finally:
        handle.remove()

    assert output.shape == value.shape
    assert seen_key_shapes == [(9, 8)]


def test_skewed_prompt_groups_stay_fully_packed(monkeypatch):
    attention = WanCrossAttention(8, 2).eval()
    value = torch.randn(1, 640, 8)
    context_bank = torch.randn(321, 1, 8)
    prompt_ids = torch.tensor([[0] * 320 + list(range(1, 321))])
    captured: dict[str, tuple[int, ...]] = {}
    original = wan_model_tools.varlen_attention

    def capture(q, k, v, **kwargs):
        captured["q"] = tuple(q.shape)
        captured["k"] = tuple(k.shape)
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(wan_model_tools, "varlen_attention", capture)
    output = attention(
        value,
        context_bank,
        torch.ones(321, dtype=torch.long),
        torch.tensor([640]),
        prompt_ids=prompt_ids,
    )

    assert output.shape == value.shape
    assert captured == {"q": (640, 2, 4), "k": (321, 2, 4)}


def test_empty_prompt_query_map_keeps_static_parameter_graph():
    attention = WanCrossAttention(8, 2)
    value = torch.randn(1, 3, 8, requires_grad=True)
    context_bank = torch.randn(1, 2, 8, requires_grad=True)
    output = attention(
        value,
        context_bank,
        torch.tensor([2]),
        torch.tensor([3]),
        query_mask=torch.zeros(1, 3, dtype=torch.bool),
        prompt_ids=torch.full((1, 3), -1, dtype=torch.long),
    )
    output.sum().backward()

    assert not output.any()
    assert value.grad is not None
    assert context_bank.grad is not None
    assert all(parameter.grad is not None for parameter in attention.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prompt_bank_flash_attention_matches_dense_token_context():
    torch.manual_seed(37)
    device = torch.device("cuda")
    dense_attention = WanCrossAttention(32, 4).to(device).eval()
    grouped_attention = WanCrossAttention(32, 4).to(device).eval()
    grouped_attention.load_state_dict(dense_attention.state_dict())
    value = torch.randn(2, 7, 32, device=device)
    context_bank = torch.randn(3, 6, 32, device=device)
    context_lens = torch.tensor([6, 4, 2], device=device)
    prompt_ids = torch.tensor(
        [[0, 0, 1, 1, 2, -1, -1], [2, 2, 0, 1, 1, 0, -1]],
        device=device,
    )
    query_lens = torch.tensor([6, 7], device=device)
    query_mask = prompt_ids >= 0
    dense_context, dense_lens = _dense_prompt_context(
        context_bank,
        context_lens,
        prompt_ids,
        query_mask,
    )

    with torch.autocast("cuda", dtype=torch.bfloat16):
        dense_output = dense_attention(
            value,
            dense_context,
            dense_lens,
            query_lens,
            query_mask=query_mask,
        )
        grouped_output = grouped_attention(
            value,
            context_bank,
            context_lens,
            query_lens,
            query_mask=query_mask,
            prompt_ids=prompt_ids,
        )

    assert_close(grouped_output, dense_output, atol=2e-2, rtol=2e-2)


def test_text_context_is_cast_to_projection_dtype_without_autocast():
    projection = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 8),
    )
    projected, lengths = embed_text_context(
        projection,
        [torch.randn(3, 8, dtype=torch.bfloat16)],
        text_len=8,
        device=torch.device("cpu"),
    )
    assert projected.dtype == next(projection.parameters()).dtype == torch.float32
    assert lengths.tolist() == [3]


def test_text_context_keeps_autocast_dtype_before_projection():
    projection = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 8),
    )
    input_dtypes = []
    handle = projection[0].register_forward_pre_hook(
        lambda _module, arguments: input_dtypes.append(arguments[0].dtype)
    )
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            projected, _ = embed_text_context(
                projection,
                [torch.randn(3, 8, dtype=torch.bfloat16)],
                text_len=8,
                device=torch.device("cpu"),
            )
    finally:
        handle.remove()
    assert input_dtypes == [torch.bfloat16]
    assert projected.dtype == torch.bfloat16


def test_vectorized_prefix_pack_preserves_row_order_and_gradients():
    source = torch.arange(3 * 5 * 2, dtype=torch.float32).reshape(3, 5, 2)
    source.requires_grad_(True)
    lengths = torch.tensor([5, 2, 4])
    packed, valid = _pack_valid_prefix(source, lengths)
    reference = torch.cat(
        [source[row, : int(length)] for row, length in enumerate(lengths)]
    )
    assert torch.equal(packed, reference)
    assert valid.tolist() == [
        [True, True, True, True, True],
        [True, True, False, False, False],
        [True, True, True, True, False],
    ]
    packed.square().sum().backward()
    assert torch.equal(
        source.grad,
        2 * source.detach() * valid[..., None],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("causal", [False, True])
def test_flash_varlen_matches_sdpa_forward_and_gradients(causal):
    torch.manual_seed(31)
    device = torch.device("cuda")
    shape = (3, 5, 2, 8)
    q_flash = torch.randn(*shape, device=device, dtype=torch.float16, requires_grad=True)
    k_flash = torch.randn(*shape, device=device, dtype=torch.float16, requires_grad=True)
    v_flash = torch.randn(*shape, device=device, dtype=torch.float16, requires_grad=True)
    q_ref = q_flash.detach().clone().requires_grad_(True)
    k_ref = k_flash.detach().clone().requires_grad_(True)
    v_ref = v_flash.detach().clone().requires_grad_(True)
    lengths = torch.tensor([5, 3, 1], device=device)

    flash = flash_attention(
        q_flash,
        k_flash,
        v_flash,
        q_lens=lengths,
        k_lens=lengths,
        causal=causal,
    )
    # Float32 forces the same public function onto its SDPA fallback.
    reference = flash_attention(
        q_ref.float(),
        k_ref.float(),
        v_ref.float(),
        q_lens=lengths,
        k_lens=lengths,
        causal=causal,
    ).to(torch.float16)
    valid = torch.arange(shape[1], device=device)[None] < lengths[:, None]
    assert torch.allclose(flash[valid], reference[valid], atol=2e-3, rtol=2e-3)

    flash.float().square().sum().backward()
    reference.float().square().sum().backward()
    for flash_grad, reference_grad in (
        (q_flash.grad, q_ref.grad),
        (k_flash.grad, k_ref.grad),
        (v_flash.grad, v_ref.grad),
    ):
        assert torch.allclose(
            flash_grad, reference_grad, atol=5e-3, rtol=5e-3
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not FLASH_ATTN_2_AVAILABLE,
    reason="CUDA FlashAttention 2 is required",
)
def test_flash_varlen_supports_zero_length_single_token_queries():
    torch.manual_seed(37)
    device = torch.device("cuda")
    q_shape = (4, 1, 2, 8)
    kv_shape = (4, 3, 2, 8)
    q_flash = torch.randn(
        *q_shape, device=device, dtype=torch.float16, requires_grad=True
    )
    k_flash = torch.randn(
        *kv_shape, device=device, dtype=torch.float16, requires_grad=True
    )
    v_flash = torch.randn(
        *kv_shape, device=device, dtype=torch.float16, requires_grad=True
    )
    q_ref = q_flash.detach().clone().float().requires_grad_(True)
    k_ref = k_flash.detach().clone().float().requires_grad_(True)
    v_ref = v_flash.detach().clone().float().requires_grad_(True)
    q_lens = torch.tensor([1, 0, 1, 0], device=device)
    k_lens = torch.tensor([3, 1, 2, 1], device=device)

    actual = flash_attention(
        q_flash,
        k_flash,
        v_flash,
        q_lens=q_lens,
        k_lens=k_lens,
    )
    reference = flash_attention(
        q_ref,
        k_ref,
        v_ref,
        q_lens=q_lens,
        k_lens=k_lens,
    )
    valid = q_lens.bool()
    assert torch.allclose(
        actual[valid].float(), reference[valid], atol=2e-3, rtol=2e-3
    )
    assert torch.count_nonzero(actual[~valid]) == 0

    actual.float().square().sum().backward()
    reference.square().sum().backward()
    for actual_grad, reference_grad in (
        (q_flash.grad, q_ref.grad),
        (k_flash.grad, k_ref.grad),
        (v_flash.grad, v_ref.grad),
    ):
        assert torch.allclose(
            actual_grad.float(), reference_grad, atol=5e-3, rtol=5e-3
        )
