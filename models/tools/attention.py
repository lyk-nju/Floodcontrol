"""Generic attention kernels used by the Floodcontrol transformer blocks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # pragma: no cover - availability depends on the runtime image
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    flash_attn = None
    FLASH_ATTN_2_AVAILABLE = False


def _length_mask(
    lengths: torch.Tensor | None,
    *,
    batch: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    if lengths is None:
        return torch.ones(batch, length, device=device, dtype=torch.bool)
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.long)
    if tuple(lengths.shape) != (batch,):
        raise ValueError(f"lengths must be [B], got {tuple(lengths.shape)}")
    return torch.arange(length, device=device)[None] < lengths[:, None]


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens: torch.Tensor | None,
    k_lens: torch.Tensor | None,
    dropout_p: float,
    softmax_scale: float | None,
    q_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    batch, q_len = q.shape[:2]
    k_len = k.shape[1]
    out_dtype = q.dtype
    q_valid = _length_mask(q_lens, batch=batch, length=q_len, device=q.device)
    k_valid = _length_mask(k_lens, batch=batch, length=k_len, device=q.device)
    allowed = q_valid[:, :, None] & k_valid[:, None, :]
    if causal:
        allowed &= torch.arange(k_len, device=q.device)[None, None, :] <= torch.arange(
            q_len, device=q.device
        )[None, :, None]
    if q_scale is not None:
        q = q * float(q_scale)
    scale = None if softmax_scale is None else float(softmax_scale)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    mask = allowed[:, None]
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=float(dropout_p),
        is_causal=False,
        scale=scale,
    )
    out = out.transpose(1, 2).contiguous().to(out_dtype)
    return out * q_valid[:, :, None, None].to(out.dtype)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    version=None,
) -> torch.Tensor:
    """Run FlashAttention when possible, otherwise use the exact SDPA fallback."""
    del version
    can_flash = (
        FLASH_ATTN_2_AVAILABLE
        and q.device.type == "cuda"
        and q.shape[-1] <= 256
        and q.dtype in (torch.float16, torch.bfloat16)
    )
    if not can_flash:
        return _sdpa_attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
        )

    batch, q_len, k_len = q.shape[0], q.shape[1], k.shape[1]
    q_lens = (
        torch.full((batch,), q_len, device=q.device, dtype=torch.int32)
        if q_lens is None
        else q_lens.to(device=q.device, dtype=torch.int32)
    )
    k_lens = (
        torch.full((batch,), k_len, device=q.device, dtype=torch.int32)
        if k_lens is None
        else k_lens.to(device=q.device, dtype=torch.int32)
    )
    q_flat = torch.cat([row[: int(length)] for row, length in zip(q, q_lens)])
    k_flat = torch.cat([row[: int(length)] for row, length in zip(k, k_lens)])
    v_flat = torch.cat([row[: int(length)] for row, length in zip(v, k_lens)])
    cu_q = torch.cat([q_lens.new_zeros(1), q_lens]).cumsum(0, dtype=torch.int32)
    cu_k = torch.cat([k_lens.new_zeros(1), k_lens]).cumsum(0, dtype=torch.int32)
    out_flat = flash_attn.flash_attn_varlen_func(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=q_len,
        max_seqlen_k=k_len,
        dropout_p=float(dropout_p),
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
    )
    out = q.new_zeros(batch, q_len, *out_flat.shape[1:])
    offset = 0
    for batch_idx, length in enumerate(q_lens.tolist()):
        out[batch_idx, :length] = out_flat[offset : offset + length]
        offset += length
    return out


def attention(*args, **kwargs) -> torch.Tensor:
    """Public generic attention entry point."""
    return flash_attention(*args, **kwargs)


__all__ = ["attention", "flash_attention"]
