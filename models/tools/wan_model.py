"""Reusable one-dimensional Wan-style transformer primitives.

The legacy video patch wrapper, ControlNet residual injection and trajectory
token branches intentionally do not live here.  RootTransformer and
BodyTransformer own their task-specific projections.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .attention import attention


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError("sinusoidal embedding dimension must be even")
    half = dim // 2
    scale = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=position.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = position.float()[..., None] * scale
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def apply_rope_with_position_ids(
    value: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Apply 1D RoPE to ``[B,L,H,D]`` using explicit integer positions."""
    if value.ndim != 4 or position_ids.ndim != 2:
        raise ValueError("RoPE expects value [B,L,H,D] and position_ids [B,L]")
    if tuple(value.shape[:2]) != tuple(position_ids.shape):
        raise ValueError("RoPE value and position_ids must share [B,L]")
    dim = value.shape[-1]
    if dim % 2:
        raise ValueError("RoPE head dimension must be even")
    inv_freq = 1.0 / (
        float(theta)
        ** (
            torch.arange(0, dim, 2, device=value.device, dtype=torch.float32)
            / float(dim)
        )
    )
    angles = position_ids.float()[..., None] * inv_freq[None, None]
    cos = angles.cos()[:, :, None]
    sin = angles.sin()[:, :, None]
    source = value.float().reshape(*value.shape[:-1], dim // 2, 2)
    first, second = source[..., 0], source[..., 1]
    rotated = torch.stack(
        [first * cos - second * sin, first * sin + second * cos], dim=-1
    ).flatten(-2)
    return rotated.to(value.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return super().forward(value.float()).to(value.dtype)


class WanSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_norm: bool = True,
        eps: float = 1e-6,
        causal: bool = False,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // num_heads
        if head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.causal = bool(causal)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(
        self,
        value: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        rope_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, dim = value.shape
        q = self.norm_q(self.q(value)).view(
            batch, length, self.num_heads, self.head_dim
        )
        k = self.norm_k(self.k(value)).view(
            batch, length, self.num_heads, self.head_dim
        )
        v = self.v(value).view(batch, length, self.num_heads, self.head_dim)
        q = apply_rope_with_position_ids(q, rope_position_ids)
        k = apply_rope_with_position_ids(k, rope_position_ids)
        out = attention(
            q,
            k,
            v,
            q_lens=seq_lens,
            k_lens=seq_lens,
            causal=self.causal,
        )
        return self.o(out.flatten(2)).to(value.dtype)


class WanCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, *, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        query_lens: torch.Tensor,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = value.shape
        if context.ndim == 4:
            if tuple(context.shape[:2]) != (batch, length):
                raise ValueError(
                    "token-aligned context must share the query [B,T] axis"
                )
            if tuple(context_lens.shape) != (batch, length):
                raise ValueError("token-aligned context_lens must be [B,T]")
            if query_mask is None:
                query_mask = (
                    torch.arange(length, device=value.device)[None]
                    < query_lens[:, None]
                )
            if tuple(query_mask.shape) != (batch, length):
                raise ValueError("query_mask must be [B,T]")
            flat_batch = batch * length
            q = self.norm_q(self.q(value)).view(
                flat_batch, 1, self.num_heads, self.head_dim
            )
            context_length = context.shape[2]
            flat_context = context.reshape(flat_batch, context_length, -1)
            k = self.norm_k(self.k(flat_context)).view(
                flat_batch, context_length, self.num_heads, self.head_dim
            )
            v = self.v(flat_context).view(
                flat_batch, context_length, self.num_heads, self.head_dim
            )
            # Invalid motion/future queries still receive a one-token dummy
            # context so both FlashAttention and SDPA remain numerically
            # defined; their projected outputs are then removed exactly.
            flat_context_lens = context_lens.reshape(-1).clamp_min(1)
            flat_query_lens = query_mask.reshape(-1).to(dtype=torch.long)
            out = attention(
                q,
                k,
                v,
                q_lens=flat_query_lens,
                k_lens=flat_context_lens,
            )
            out = self.o(out.flatten(2)).reshape(batch, length, -1)
            return out.to(value.dtype) * query_mask[..., None].to(value.dtype)

        if context.ndim != 3:
            raise ValueError("context must be [B,L,D] or token-aligned [B,T,L,D]")
        q = self.norm_q(self.q(value)).view(
            batch, length, self.num_heads, self.head_dim
        )
        k = self.norm_k(self.k(context)).view(
            batch, context.shape[1], self.num_heads, self.head_dim
        )
        v = self.v(context).view(
            batch, context.shape[1], self.num_heads, self.head_dim
        )
        out = attention(q, k, v, q_lens=query_lens, k_lens=context_lens)
        out = self.o(out.flatten(2)).to(value.dtype)
        if query_mask is not None:
            if tuple(query_mask.shape) != (batch, length):
                raise ValueError("query_mask must be [B,T]")
            out = out * query_mask[..., None].to(out.dtype)
        return out


class WanTransformerBlock(nn.Module):
    """Non-causal Wan-style block with per-token diffusion modulation."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        *,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        causal: bool = False,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.self_attn = WanSelfAttention(
            dim, num_heads, qk_norm=qk_norm, eps=eps, causal=causal
        )
        self.cross_attn = WanCrossAttention(
            dim, num_heads, qk_norm=qk_norm, eps=eps
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        value: torch.Tensor,
        *,
        modulation: torch.Tensor,
        seq_lens: torch.Tensor,
        rope_position_ids: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        text_query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if modulation.shape != (*value.shape[:2], 6, value.shape[-1]):
            raise ValueError("modulation must be [B,L,6,D]")
        pieces = (modulation.float() + self.modulation).chunk(6, dim=2)
        normalized = self.norm1(value).float() * (1 + pieces[1].squeeze(2))
        normalized = normalized + pieces[0].squeeze(2)
        value = value + self.self_attn(
            normalized.to(value.dtype),
            seq_lens=seq_lens,
            rope_position_ids=rope_position_ids,
        ) * pieces[2].squeeze(2).to(value.dtype)
        value = value + self.cross_attn(
            self.norm3(value),
            context,
            context_lens,
            seq_lens,
            query_mask=text_query_mask,
        )
        normalized = self.norm2(value).float() * (1 + pieces[4].squeeze(2))
        normalized = normalized + pieces[3].squeeze(2)
        value = value + self.ffn(normalized.to(value.dtype)) * pieces[5].squeeze(2).to(
            value.dtype
        )
        return value


def project_unique_text_context(
    projection: nn.Module,
    context: list[torch.Tensor],
    *,
    text_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project unique tensor identities and return per-entry prompt IDs."""
    if not context:
        raise ValueError("text context cannot be empty")
    projection_parameter = next(projection.parameters(), None)
    if projection_parameter is None:
        raise ValueError("text projection must own at least one parameter")
    # Under mixed-precision autocast, retain the stored BF16/FP16 text dtype
    # instead of materializing an FP32 [N,L,4096] temporary. Outside autocast,
    # match the projection parameters so evaluation and Web inference remain
    # dtype-safe.
    projection_dtype = (
        torch.get_autocast_dtype(device.type)
        if torch.is_autocast_enabled(device.type)
        else projection_parameter.dtype
    )
    unique: list[torch.Tensor] = []
    unique_by_identity: dict[int, int] = {}
    gather_indices: list[int] = []
    for item in context:
        identity = id(item)
        if identity not in unique_by_identity:
            unique_by_identity[identity] = len(unique)
            unique.append(item)
        gather_indices.append(unique_by_identity[identity])
    unique_lengths_list = [
        min(int(item.shape[0]), int(text_len)) for item in unique
    ]
    padded_length = max(1, max(unique_lengths_list))
    padded_unique = torch.stack(
        [
            torch.cat(
                [
                    item[:padded_length].to(
                        device=device,
                        dtype=projection_dtype,
                    ),
                    torch.zeros(
                        max(0, padded_length - item.shape[0]),
                        item.shape[-1],
                        device=device,
                        dtype=projection_dtype,
                    ),
                ],
                dim=0,
            )
            for item in unique
        ],
        dim=0,
    )
    projected_unique = projection(padded_unique)
    unique_lengths = torch.tensor(
        unique_lengths_list, device=device, dtype=torch.long
    )
    indices = torch.tensor(gather_indices, device=device, dtype=torch.long)
    return projected_unique, unique_lengths, indices


def embed_text_context(
    projection: nn.Module,
    context: list[torch.Tensor],
    *,
    text_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad and project one text feature sequence per timeline entry."""

    projected, unique_lengths, prompt_ids = project_unique_text_context(
        projection,
        context,
        text_len=text_len,
        device=device,
    )
    return (
        projected.index_select(0, prompt_ids),
        unique_lengths.index_select(0, prompt_ids),
    )


__all__ = [
    "WanCrossAttention",
    "WanLayerNorm",
    "WanRMSNorm",
    "WanSelfAttention",
    "WanTransformerBlock",
    "apply_rope_with_position_ids",
    "embed_text_context",
    "project_unique_text_context",
    "sinusoidal_embedding_1d",
]
