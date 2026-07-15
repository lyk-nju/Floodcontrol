"""Typed contracts and pure condition helpers for the hybrid LDF.

This module deliberately owns no learnable parameters.  It is the single
boundary between absolute/runtime conditions and the tensors consumed by the
Root/Body transformers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping

import torch

from utils.motion_process import LOCAL_ROOT_DIM, ROOT_DIM
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
    require_aligned_frame_count,
    token_count_to_frame_count,
    token_index_to_frame_start,
)


def _require_tensor(name: str, value: torch.Tensor, ndim: int) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {tuple(value.shape)}")


def _check_pair(name: str, value: torch.Tensor | None, mask: torch.Tensor | None) -> None:
    if (value is None) != (mask is None):
        raise ValueError(f"{name} value and mask must either both be set or both be None")
    if value is not None:
        if tuple(value.shape) != tuple(mask.shape):
            raise ValueError(
                f"{name} value/mask shapes differ: {tuple(value.shape)} != {tuple(mask.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} mask must be bool, got {mask.dtype}")


def _is_prefix_mask(mask: torch.Tensor) -> bool:
    """Return whether every row is True* followed by False*."""
    if mask.shape[-1] <= 1:
        return True
    return not bool((~mask[..., :-1] & mask[..., 1:]).any())


@dataclass(frozen=True)
class HybridMotion:
    """The only persistent generative state: explicit root plus body latent."""

    root_motion: torch.Tensor
    latent_motion: torch.Tensor

    def validate(self) -> None:
        _require_tensor("root_motion", self.root_motion, 4)
        _require_tensor("latent_motion", self.latent_motion, 3)
        if tuple(self.root_motion.shape[2:]) != (FRAMES_PER_TOKEN, ROOT_DIM):
            raise ValueError(
                "root_motion must be [B,T,4,5], "
                f"got {tuple(self.root_motion.shape)}"
            )
        if tuple(self.root_motion.shape[:2]) != tuple(self.latent_motion.shape[:2]):
            raise ValueError("root_motion and latent_motion must share [B,T]")
        if self.root_motion.device != self.latent_motion.device:
            raise ValueError("root_motion and latent_motion must share a device")
        if not self.root_motion.is_floating_point() or not self.latent_motion.is_floating_point():
            raise TypeError("HybridMotion values must be floating point")

    @property
    def batch_size(self) -> int:
        return int(self.root_motion.shape[0])

    @property
    def token_length(self) -> int:
        return int(self.root_motion.shape[1])

    def clone(self, *, detach: bool = False) -> "HybridMotion":
        root = self.root_motion.detach() if detach else self.root_motion
        latent = self.latent_motion.detach() if detach else self.latent_motion
        return HybridMotion(root.clone(), latent.clone())


@dataclass(frozen=True)
class LDFCondition:
    """Prepared text/current/future observations consumed by :class:`LDF`."""

    text_context: list[torch.Tensor]
    text_null_context: list[torch.Tensor]
    root_condition_value: torch.Tensor | None = None
    root_condition_mask: torch.Tensor | None = None
    body_condition_value: torch.Tensor | None = None
    body_condition_mask: torch.Tensor | None = None
    future_root_condition_value: torch.Tensor | None = None
    future_root_condition_mask: torch.Tensor | None = None
    future_timeline_position_ids: torch.Tensor | None = None
    future_valid_mask: torch.Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int | None = None,
        token_length: int | None = None,
        latent_dim: int | None = None,
    ) -> None:
        if not isinstance(self.text_context, list) or not isinstance(
            self.text_null_context, list
        ):
            raise TypeError("text_context and text_null_context must be lists")
        if batch_size is not None:
            valid_text_lengths = {batch_size}
            if token_length is not None:
                valid_text_lengths.add(batch_size * token_length)
            if len(self.text_context) not in valid_text_lengths:
                raise ValueError(
                    f"text_context length {len(self.text_context)} is not one of "
                    f"{sorted(valid_text_lengths)}"
                )
            if len(self.text_null_context) != batch_size:
                raise ValueError("text_null_context must contain one tensor per sample")

        _check_pair(
            "root_condition", self.root_condition_value, self.root_condition_mask
        )
        _check_pair(
            "body_condition", self.body_condition_value, self.body_condition_mask
        )
        _check_pair(
            "future_root_condition",
            self.future_root_condition_value,
            self.future_root_condition_mask,
        )

        if self.root_condition_value is not None:
            _require_tensor("root_condition_value", self.root_condition_value, 4)
            if tuple(self.root_condition_value.shape[2:]) != (
                FRAMES_PER_TOKEN,
                ROOT_DIM,
            ):
                raise ValueError("root_condition_value must be [B,T,4,5]")
            if token_length is not None and self.root_condition_value.shape[1] != token_length:
                raise ValueError("root_condition token length does not match LDF input")
            heading_mask = self.root_condition_mask[..., 3:5]
            if bool((heading_mask[..., 0] != heading_mask[..., 1]).any()):
                raise ValueError("heading cos/sin constraints must always be masked together")

        if self.body_condition_value is not None:
            _require_tensor("body_condition_value", self.body_condition_value, 3)
            if token_length is not None and self.body_condition_value.shape[1] != token_length:
                raise ValueError("body_condition token length does not match LDF input")
            if latent_dim is not None and self.body_condition_value.shape[-1] != latent_dim:
                raise ValueError("body_condition feature dimension does not match latent_motion")

        future_fields = (
            self.future_root_condition_value,
            self.future_root_condition_mask,
            self.future_timeline_position_ids,
            self.future_valid_mask,
        )
        if any(value is not None for value in future_fields) and not all(
            value is not None for value in future_fields
        ):
            raise ValueError("all future root fields must be supplied together")
        if self.future_root_condition_value is not None:
            _require_tensor(
                "future_root_condition_value", self.future_root_condition_value, 4
            )
            _require_tensor(
                "future_timeline_position_ids",
                self.future_timeline_position_ids,
                2,
            )
            _require_tensor("future_valid_mask", self.future_valid_mask, 2)
            if tuple(self.future_root_condition_value.shape[2:]) != (
                FRAMES_PER_TOKEN,
                ROOT_DIM,
            ):
                raise ValueError("future_root_condition_value must be [B,N,4,5]")
            prefix = tuple(self.future_root_condition_value.shape[:2])
            if tuple(self.future_timeline_position_ids.shape) != prefix or tuple(
                self.future_valid_mask.shape
            ) != prefix:
                raise ValueError("future root fields must share [B,N]")
            if self.future_timeline_position_ids.dtype != torch.long:
                raise TypeError("future_timeline_position_ids must be int64")
            if self.future_valid_mask.dtype != torch.bool:
                raise TypeError("future_valid_mask must be bool")
            if not _is_prefix_mask(self.future_valid_mask):
                raise ValueError("future_valid_mask must be prefix-valid for packed attention")
            for positions, valid in zip(
                self.future_timeline_position_ids, self.future_valid_mask
            ):
                valid_positions = positions[valid]
                if valid_positions.numel() > 1 and bool(
                    (valid_positions[1:] <= valid_positions[:-1]).any()
                ):
                    raise ValueError(
                        "valid future timeline positions must be strictly increasing"
                    )
            heading_mask = self.future_root_condition_mask[..., 3:5]
            if bool((heading_mask[..., 0] != heading_mask[..., 1]).any()):
                raise ValueError("future heading cos/sin constraints must be masked together")

    def clone(self) -> "LDFCondition":
        def clone_value(value):
            if torch.is_tensor(value):
                return value.clone()
            if isinstance(value, list):
                return [clone_value(item) for item in value]
            return copy.deepcopy(value)

        return LDFCondition(**{name: clone_value(getattr(self, name)) for name in self.__dataclass_fields__})


@dataclass(frozen=True)
class LDFInput:
    noisy_motion: HybridMotion
    beta: torch.Tensor
    history_mask: torch.Tensor
    generation_mask: torch.Tensor
    timeline_position_ids: torch.Tensor
    rope_position_ids: torch.Tensor
    previous_root_frame: torch.Tensor | None
    condition: LDFCondition

    def validate(self) -> None:
        self.noisy_motion.validate()
        batch, tokens = self.noisy_motion.root_motion.shape[:2]
        for name, value, dtype in (
            ("beta", self.beta, None),
            ("history_mask", self.history_mask, torch.bool),
            ("generation_mask", self.generation_mask, torch.bool),
            ("timeline_position_ids", self.timeline_position_ids, torch.long),
            ("rope_position_ids", self.rope_position_ids, torch.long),
        ):
            _require_tensor(name, value, 2)
            if tuple(value.shape) != (batch, tokens):
                raise ValueError(f"{name} must be [B,T]")
            if dtype is not None and value.dtype != dtype:
                raise TypeError(f"{name} must have dtype {dtype}")
        if not self.beta.is_floating_point():
            raise TypeError("beta must be floating point")
        if bool(((self.beta < 0) | (self.beta > 1)).any()):
            raise ValueError("beta must lie in [0,1]")
        if bool((self.history_mask & self.generation_mask).any()):
            raise ValueError("history_mask and generation_mask must be disjoint")
        valid = self.history_mask | self.generation_mask
        if not _is_prefix_mask(valid):
            raise ValueError("valid motion tokens must form a prefix")
        if bool((self.history_mask & (self.beta.abs() > 1e-6)).any()):
            raise ValueError("history tokens must be clean (beta=0)")
        if tokens > 1 and bool(
            (
                self.timeline_position_ids[:, 1:]
                != self.timeline_position_ids[:, :-1] + 1
            ).any()
        ):
            raise ValueError("motion timeline positions must be contiguous")
        rope_origin = self.timeline_position_ids - self.rope_position_ids
        if bool((rope_origin != rope_origin[:, :1]).any()):
            raise ValueError(
                "timeline_position_ids and rope_position_ids must differ by one "
                "constant origin per sample"
            )
        has_generation = self.generation_mask.any(dim=1)
        if bool(has_generation.any()):
            first_generation = self.generation_mask.to(torch.int64).argmax(dim=1)
            first_generation_rope = self.rope_position_ids.gather(
                1, first_generation[:, None]
            ).squeeze(1)
            if bool((first_generation_rope[has_generation] != 0).any()):
                raise ValueError("the first generation token must have RoPE position 0")
        if self.previous_root_frame is not None:
            _require_tensor("previous_root_frame", self.previous_root_frame, 2)
            if tuple(self.previous_root_frame.shape) != (batch, ROOT_DIM):
                raise ValueError("previous_root_frame must be [B,5]")
        self.condition.validate(
            batch_size=batch,
            token_length=tokens,
            latent_dim=self.noisy_motion.latent_motion.shape[-1],
        )
        future_positions = self.condition.future_timeline_position_ids
        future_valid = self.condition.future_valid_mask
        if future_positions is not None:
            for batch_idx in range(batch):
                valid_current = valid[batch_idx]
                valid_future = future_valid[batch_idx]
                if bool(valid_current.any()) and bool(valid_future.any()):
                    current_end = self.timeline_position_ids[batch_idx, valid_current][-1]
                    if bool(
                        (future_positions[batch_idx, valid_future] <= current_end).any()
                    ):
                        raise ValueError(
                            "future timeline positions must lie after the current motion window"
                        )

    @property
    def rope_origin(self) -> torch.Tensor:
        """Absolute timeline position represented by RoPE position zero."""
        return self.timeline_position_ids[:, :1] - self.rope_position_ids[:, :1]

    def timeline_to_rope(self, timeline_position_ids: torch.Tensor) -> torch.Tensor:
        """Convert absolute timeline IDs to this input's generation-centered RoPE IDs."""
        _require_tensor("timeline_position_ids", timeline_position_ids, 2)
        if timeline_position_ids.shape[0] != self.noisy_motion.batch_size:
            raise ValueError("timeline_position_ids batch size does not match LDFInput")
        if timeline_position_ids.dtype != torch.long:
            raise TypeError("timeline_position_ids must be int64")
        return timeline_position_ids - self.rope_origin.to(timeline_position_ids.device)


@dataclass(frozen=True)
class LDFPrediction:
    velocity: HybridMotion
    clean_root_motion: torch.Tensor
    local_root_motion: torch.Tensor
    local_root_feature_valid: torch.Tensor


@dataclass(frozen=True)
class LDFStreamState:
    noisy_motion: HybridMotion
    current_step: int
    commit_index: int
    window_origin: int
    epoch: int
    previous_root_frame: torch.Tensor | None
    rng_state: torch.Tensor
    num_denoise_steps: int

    def validate(self) -> None:
        self.noisy_motion.validate()
        if self.current_step < 0 or self.commit_index < 0 or self.window_origin < 0:
            raise ValueError("stream indices must be non-negative")
        if self.num_denoise_steps <= 0:
            raise ValueError("num_denoise_steps must be positive")
        if self.previous_root_frame is not None:
            if tuple(self.previous_root_frame.shape) != (
                self.noisy_motion.batch_size,
                ROOT_DIM,
            ):
                raise ValueError("stream previous_root_frame must be [B,5]")


def normalize_features(
    values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (values - mean.to(values)) / std.to(values).clamp_min(1e-8)


def unnormalize_features(
    values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return values * std.to(values) + mean.to(values)


def _slice_and_pad(
    value: torch.Tensor | None,
    start: int,
    length: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    sliced = value[:, start : start + length]
    missing = length - sliced.shape[1]
    if missing > 0:
        sliced = torch.cat(
            [sliced, sliced.new_zeros(sliced.shape[0], missing, *sliced.shape[2:])],
            dim=1,
        )
    return sliced


def create_window_condition(
    *,
    text_context: list[torch.Tensor],
    text_null_context: list[torch.Tensor],
    window_origin: int,
    window_tokens: int,
    future_tokens: int = 0,
    root_condition_value: torch.Tensor | None = None,
    root_condition_mask: torch.Tensor | None = None,
    body_condition_value: torch.Tensor | None = None,
    body_condition_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Slice dense absolute condition timelines to one active token window."""
    if window_origin < 0 or window_tokens <= 0 or future_tokens < 0:
        raise ValueError("invalid window bounds")
    _check_pair("root_condition", root_condition_value, root_condition_mask)
    _check_pair("body_condition", body_condition_value, body_condition_mask)
    frame_start = token_index_to_frame_start(window_origin)
    frame_count = token_count_to_frame_count(window_tokens)
    future_start = frame_start + frame_count
    future_count = token_count_to_frame_count(future_tokens)
    return {
        "text_context": list(text_context),
        "text_null_context": list(text_null_context),
        "root_condition_value": _slice_and_pad(
            root_condition_value, frame_start, frame_count
        ),
        "root_condition_mask": _slice_and_pad(
            root_condition_mask, frame_start, frame_count
        ),
        "body_condition_value": _slice_and_pad(
            body_condition_value, window_origin, window_tokens
        ),
        "body_condition_mask": _slice_and_pad(
            body_condition_mask, window_origin, window_tokens
        ),
        "future_root_condition_value": _slice_and_pad(
            root_condition_value, future_start, future_count
        ),
        "future_root_condition_mask": _slice_and_pad(
            root_condition_mask, future_start, future_count
        ),
        "future_timeline_position_ids": torch.arange(
            window_origin + window_tokens,
            window_origin + window_tokens + future_tokens,
            device=(root_condition_value.device if root_condition_value is not None else None),
            dtype=torch.long,
        ),
    }


def _root_to_tokens(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    if value.ndim == 4:
        return value
    if value.ndim != 3 or value.shape[-1] != ROOT_DIM:
        raise ValueError("root condition must be [B,F,5] or [B,T,4,5]")
    frames = require_aligned_frame_count(value.shape[1])
    tokens = frame_count_to_token_count(frames)
    return value.reshape(value.shape[0], tokens, FRAMES_PER_TOKEN, ROOT_DIM)


def create_ldf_condition(window_condition: Mapping[str, Any]) -> LDFCondition:
    """Compile a window mapping into the strict model-facing condition."""
    root_value = _root_to_tokens(window_condition.get("root_condition_value"))
    root_mask = _root_to_tokens(window_condition.get("root_condition_mask"))
    future_value = _root_to_tokens(
        window_condition.get("future_root_condition_value")
    )
    future_mask = _root_to_tokens(window_condition.get("future_root_condition_mask"))
    future_valid = None
    future_timeline_position_ids = None
    if future_value is not None:
        future_valid = future_mask.flatten(2).any(dim=-1)
        # Pack valid future tokens to the left, keeping value/mask/position aligned.
        raw_pos = window_condition.get("future_timeline_position_ids")
        if raw_pos is None:
            raw_pos = torch.arange(
                future_value.shape[1], device=future_value.device, dtype=torch.long
            )
        if raw_pos.ndim == 1:
            raw_pos = raw_pos[None].expand(future_value.shape[0], -1)
        packed_value = torch.zeros_like(future_value)
        packed_mask = torch.zeros_like(future_mask)
        packed_pos = torch.zeros_like(raw_pos, dtype=torch.long)
        packed_valid = torch.zeros_like(future_valid)
        for batch_idx in range(future_value.shape[0]):
            indices = torch.nonzero(future_valid[batch_idx], as_tuple=False).flatten()
            count = int(indices.numel())
            if count:
                packed_value[batch_idx, :count] = future_value[batch_idx, indices]
                packed_mask[batch_idx, :count] = future_mask[batch_idx, indices]
                packed_pos[batch_idx, :count] = raw_pos[batch_idx, indices]
                packed_valid[batch_idx, :count] = True
        future_value, future_mask = packed_value, packed_mask
        future_timeline_position_ids, future_valid = packed_pos, packed_valid

    condition = LDFCondition(
        text_context=list(window_condition.get("text_context", [])),
        text_null_context=list(window_condition.get("text_null_context", [])),
        root_condition_value=root_value,
        root_condition_mask=root_mask,
        body_condition_value=window_condition.get("body_condition_value"),
        body_condition_mask=window_condition.get("body_condition_mask"),
        future_root_condition_value=future_value,
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=future_timeline_position_ids,
        future_valid_mask=future_valid,
    )
    condition.validate()
    return condition


def _without_constraints(condition: LDFCondition) -> LDFCondition:
    def clear_mask(mask):
        return None if mask is None else torch.zeros_like(mask, dtype=torch.bool)

    return replace(
        condition,
        root_condition_mask=clear_mask(condition.root_condition_mask),
        body_condition_mask=clear_mask(condition.body_condition_mask),
        future_root_condition_mask=clear_mask(condition.future_root_condition_mask),
        future_valid_mask=(
            None
            if condition.future_valid_mask is None
            else torch.zeros_like(condition.future_valid_mask, dtype=torch.bool)
        ),
    )


def create_cfg_condition(condition: LDFCondition) -> dict[str, LDFCondition]:
    """Create joint/text/constraint/history branches without mutating input."""
    history = _without_constraints(condition)
    history = replace(history, text_context=list(condition.text_null_context))
    text_only = _without_constraints(condition)
    constraint_only = replace(condition, text_context=list(condition.text_null_context))
    return {
        "joint": condition.clone(),
        "text": text_only,
        "constraint": constraint_only,
        "history": history,
    }


__all__ = [
    "ROOT_DIM",
    "LOCAL_ROOT_DIM",
    "HybridMotion",
    "LDFCondition",
    "LDFInput",
    "LDFPrediction",
    "LDFStreamState",
    "create_cfg_condition",
    "create_ldf_condition",
    "create_window_condition",
    "normalize_features",
    "unnormalize_features",
]
