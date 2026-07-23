"""Typed contracts and pure condition helpers for the hybrid LDF.

This module deliberately owns no learnable parameters.  It is the single
boundary between absolute/runtime conditions and the tensors consumed by the
Root/Body transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import torch

from utils.motion_process import LOCAL_ROOT_DIM, ROOT_DIM
from utils.token_frame import (
    FRAMES_PER_TOKEN,
    frame_count_to_token_count,
    require_aligned_frame_count,
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
    """Persistent state: physical root5 plus raw deterministic VAE latent."""

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
    future_horizon_tokens: torch.Tensor | None = None

    def validate_structure(
        self,
        *,
        batch_size: int | None = None,
        token_length: int | None = None,
        latent_dim: int | None = None,
    ) -> None:
        """Validate tensor contracts without inspecting accelerator contents."""

        if not isinstance(self.text_context, list) or not isinstance(
            self.text_null_context, list
        ):
            raise TypeError("text_context and text_null_context must be lists")
        text_dim = None
        for index, value in enumerate(self.text_context + self.text_null_context):
            if not torch.is_tensor(value) or value.ndim != 2:
                raise ValueError(
                    f"text context {index} must be a rank-2 tensor [L,C]"
                )
            if not value.is_floating_point():
                raise ValueError("text contexts must contain floating-point values")
            if value.shape[0] <= 0 or value.shape[1] <= 0:
                raise ValueError(
                    "text contexts must have positive sequence and feature sizes"
                )
            if text_dim is None:
                text_dim = int(value.shape[1])
            elif value.shape[1] != text_dim:
                raise ValueError("all text contexts must share one feature dimension")
        if batch_size is not None:
            if token_length is None:
                raise ValueError(
                    "token_length is required when validating conditional text"
                )
            expected_text_length = batch_size * token_length
            if len(self.text_context) != expected_text_length:
                raise ValueError(
                    "text_context must contain one tensor per motion token: "
                    f"expected B*T={expected_text_length}, got {len(self.text_context)}"
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
            if batch_size is not None and self.root_condition_value.shape[0] != batch_size:
                raise ValueError("root_condition batch size does not match LDF input")
            if token_length is not None and self.root_condition_value.shape[1] != token_length:
                raise ValueError("root_condition token length does not match LDF input")
        if self.body_condition_value is not None:
            _require_tensor("body_condition_value", self.body_condition_value, 3)
            if batch_size is not None and self.body_condition_value.shape[0] != batch_size:
                raise ValueError("body_condition batch size does not match LDF input")
            if token_length is not None and self.body_condition_value.shape[1] != token_length:
                raise ValueError("body_condition token length does not match LDF input")
            if latent_dim is not None and self.body_condition_value.shape[-1] != latent_dim:
                raise ValueError(
                    "body_condition feature dimension does not match latent_motion"
                )

        future_fields = (
            self.future_root_condition_value,
            self.future_root_condition_mask,
            self.future_timeline_position_ids,
            self.future_valid_mask,
            self.future_horizon_tokens,
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
            _require_tensor(
                "future_horizon_tokens", self.future_horizon_tokens, 1
            )
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
            if batch_size is not None and prefix[0] != batch_size:
                raise ValueError("future root batch size does not match LDF input")
            if tuple(self.future_horizon_tokens.shape) != (prefix[0],):
                raise ValueError("future_horizon_tokens must be [B]")
            if self.future_timeline_position_ids.dtype != torch.long:
                raise TypeError("future_timeline_position_ids must be int64")
            if self.future_valid_mask.dtype != torch.bool:
                raise TypeError("future_valid_mask must be bool")
            if self.future_horizon_tokens.dtype != torch.long:
                raise TypeError("future_horizon_tokens must be int64")

    def validate(
        self,
        *,
        batch_size: int | None = None,
        token_length: int | None = None,
        latent_dim: int | None = None,
    ) -> None:
        self.validate_structure(
            batch_size=batch_size,
            token_length=token_length,
            latent_dim=latent_dim,
        )
        finite_checked: set[int] = set()
        for value in self.text_context + self.text_null_context:
            identity = id(value)
            if identity not in finite_checked:
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("text contexts must contain finite values")
                finite_checked.add(identity)

        for name, value in (
            ("root_condition_value", self.root_condition_value),
            ("body_condition_value", self.body_condition_value),
            ("future_root_condition_value", self.future_root_condition_value),
        ):
            if value is not None and not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite values")

        if self.root_condition_value is not None:
            heading_mask = self.root_condition_mask[..., 3:5]
            if bool((heading_mask[..., 0] != heading_mask[..., 1]).any()):
                raise ValueError("heading cos/sin constraints must always be masked together")
        if self.future_root_condition_value is not None:
            if bool((self.future_horizon_tokens < 0).any()):
                raise ValueError("future_horizon_tokens must be non-negative")
            if not _is_prefix_mask(self.future_valid_mask):
                raise ValueError("future_valid_mask must be prefix-valid for packed attention")
            mask_valid = self.future_root_condition_mask.flatten(2).any(dim=-1)
            if not torch.equal(mask_valid, self.future_valid_mask):
                raise ValueError(
                    "future_valid_mask must exactly match constrained future tokens"
                )
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

@dataclass(frozen=True)
class LDFInput:
    noisy_motion: HybridMotion
    beta: torch.Tensor
    history_mask: torch.Tensor
    generation_mask: torch.Tensor
    timeline_position_ids: torch.Tensor
    rope_position_ids: torch.Tensor
    previous_root_frame: torch.Tensor | None
    previous_root_valid_mask: torch.Tensor | None
    condition: LDFCondition

    def validate_structure(self) -> None:
        """Validate ranks, shapes and dtypes without synchronizing tensor values."""

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
            if value.device != self.noisy_motion.root_motion.device:
                raise ValueError(f"{name} must share the motion device")
        if not self.beta.is_floating_point():
            raise TypeError("beta must be floating point")
        if (self.previous_root_frame is None) != (
            self.previous_root_valid_mask is None
        ):
            raise ValueError(
                "previous_root_frame and previous_root_valid_mask must both be set or both be None"
            )
        if self.previous_root_frame is not None:
            _require_tensor("previous_root_frame", self.previous_root_frame, 2)
            if tuple(self.previous_root_frame.shape) != (batch, ROOT_DIM):
                raise ValueError("previous_root_frame must be [B,5]")
            _require_tensor(
                "previous_root_valid_mask", self.previous_root_valid_mask, 1
            )
            if tuple(self.previous_root_valid_mask.shape) != (batch,):
                raise ValueError("previous_root_valid_mask must be [B]")
            if self.previous_root_valid_mask.dtype != torch.bool:
                raise TypeError("previous_root_valid_mask must be bool")
            if self.previous_root_frame.device != self.noisy_motion.root_motion.device:
                raise ValueError("previous_root_frame must share the motion device")
            if self.previous_root_valid_mask.device != self.noisy_motion.root_motion.device:
                raise ValueError("previous_root_valid_mask must share the motion device")
        self.condition.validate_structure(
            batch_size=batch,
            token_length=tokens,
            latent_dim=self.noisy_motion.latent_motion.shape[-1],
        )

    def validate(self) -> None:
        self.validate_structure()
        batch, tokens = self.noisy_motion.root_motion.shape[:2]
        for name, value in (
            ("root_motion", self.noisy_motion.root_motion),
            ("latent_motion", self.noisy_motion.latent_motion),
            ("beta", self.beta),
        ):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite values")
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
            if not bool(torch.isfinite(self.previous_root_frame).all()):
                raise ValueError("previous_root_frame must contain finite values")
        self.condition.validate(
            batch_size=batch,
            token_length=tokens,
            latent_dim=self.noisy_motion.latent_motion.shape[-1],
        )
        future_positions = self.condition.future_timeline_position_ids
        future_valid = self.future_attention_mask()
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

    def future_attention_mask(self) -> torch.Tensor | None:
        """Select the future constraints visible at this denoising microstep.

        The condition stores one immutable, prefix-packed candidate superset for
        the whole commit.  Motion visibility changes with the triangular
        schedule, so this method removes candidates which have already become
        motion queries, applies the configured absolute lookahead, and caps the
        combined motion/future query count at the model window length.
        """

        condition = self.condition
        if condition.future_valid_mask is None:
            return None
        visible_motion = self.history_mask | self.generation_mask
        visible_count = visible_motion.sum(dim=1, dtype=torch.long)
        visible_end = self.timeline_position_ids[:, 0] + visible_count
        future_positions = condition.future_timeline_position_ids.to(
            device=self.timeline_position_ids.device
        )
        future_valid = condition.future_valid_mask.to(
            device=self.timeline_position_ids.device
        )
        horizon = condition.future_horizon_tokens.to(
            device=self.timeline_position_ids.device
        )
        selected = (
            future_valid
            & (future_positions >= visible_end[:, None])
            & (future_positions < visible_end[:, None] + horizon[:, None])
        )

        # Sparse waypoint/goal candidates need a rank-based capacity limit.  A
        # positional cutoff could otherwise under-fill the available budget.
        available_slots = self.noisy_motion.token_length - visible_count
        selected_rank = selected.cumsum(dim=1, dtype=torch.long)
        return selected & (selected_rank <= available_slots[:, None])


@dataclass(frozen=True)
class LDFPrediction:
    """Native outputs plus physical and solver-facing interpretations.

    ``clean_motion.root_motion`` is always the projected physical Root-to-Body
    view. The Root solver endpoint remains raw until commit and is represented
    only through ``solver_velocity``.
    """

    raw_root_output: torch.Tensor
    raw_body_output: torch.Tensor
    clean_motion: HybridMotion
    solver_velocity: HybridMotion
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


def create_ldf_condition(
    *,
    text_context: list[torch.Tensor],
    text_null_context: list[torch.Tensor],
    root_condition_value: torch.Tensor | None = None,
    root_condition_mask: torch.Tensor | None = None,
    body_condition_value: torch.Tensor | None = None,
    body_condition_mask: torch.Tensor | None = None,
    future_root_condition_value: torch.Tensor | None = None,
    future_root_condition_mask: torch.Tensor | None = None,
    future_timeline_position_ids: torch.Tensor | None = None,
    future_horizon_tokens: torch.Tensor | int | None = None,
    validate_numerics: bool = True,
) -> LDFCondition:
    """Compile source-specific candidates into the sole model-facing condition.

    Training supplies sampled tensors and runtime supplies route/observation
    tensors, but both cross the same shape conversion and future packing
    boundary here.  Dynamic Future is intentionally not inferred from a fixed
    window end; callers provide an absolute candidate superset and
    :class:`LDFInput` decides which candidates remain beyond visible motion.
    """

    _check_pair("root_condition", root_condition_value, root_condition_mask)
    _check_pair("body_condition", body_condition_value, body_condition_mask)
    _check_pair(
        "future_root_condition",
        future_root_condition_value,
        future_root_condition_mask,
    )
    batch_size = len(text_null_context)
    if batch_size <= 0:
        raise ValueError("text_null_context must contain one tensor per sample")
    if len(text_context) == 0 or len(text_context) % batch_size:
        raise ValueError("text_context must contain a non-empty B*T timeline")
    token_length = len(text_context) // batch_size

    root_value = _root_to_tokens(root_condition_value)
    root_mask = _root_to_tokens(root_condition_mask)
    future_value = _root_to_tokens(future_root_condition_value)
    future_mask = _root_to_tokens(future_root_condition_mask)
    raw_future_positions = future_timeline_position_ids
    raw_future_horizon = future_horizon_tokens
    if future_value is not None and future_value.shape[1] == 0:
        future_value = None
        future_mask = None
    future_valid = None
    future_timeline_position_ids = None
    if future_value is not None:
        future_valid = future_mask.flatten(2).any(dim=-1)
        packed_tokens = int(future_valid.sum(dim=1).max().item())
        if packed_tokens == 0:
            future_value = future_mask = future_valid = None
        else:
            # Pack valid future tokens to a compact prefix while preserving
            # value/mask/absolute-position alignment.
            raw_pos = raw_future_positions
            if raw_pos is None:
                raise ValueError(
                    "future_timeline_position_ids is required with future root conditions"
                )
            raw_pos = torch.as_tensor(
                raw_pos,
                device=future_value.device,
                dtype=torch.long,
            )
            if raw_pos.ndim == 1:
                raw_pos = raw_pos[None].expand(future_value.shape[0], -1)
            if tuple(raw_pos.shape) != tuple(future_value.shape[:2]):
                raise ValueError(
                    "future_timeline_position_ids must match future root [B,N]"
                )
            packed_value = future_value.new_zeros(
                future_value.shape[0], packed_tokens, *future_value.shape[2:]
            )
            packed_mask = torch.zeros_like(packed_value, dtype=torch.bool)
            packed_pos = raw_pos.new_zeros(raw_pos.shape[0], packed_tokens)
            packed_valid = torch.zeros(
                raw_pos.shape[0],
                packed_tokens,
                device=future_value.device,
                dtype=torch.bool,
            )
            for batch_idx in range(future_value.shape[0]):
                indices = torch.nonzero(
                    future_valid[batch_idx], as_tuple=False
                ).flatten()
                count = int(indices.numel())
                if count:
                    packed_value[batch_idx, :count] = future_value[batch_idx, indices]
                    packed_mask[batch_idx, :count] = future_mask[batch_idx, indices]
                    packed_pos[batch_idx, :count] = raw_pos[batch_idx, indices]
                    packed_valid[batch_idx, :count] = True
            future_value, future_mask = packed_value, packed_mask
            future_timeline_position_ids, future_valid = packed_pos, packed_valid

    future_horizon_tokens = None
    if future_value is not None:
        if raw_future_horizon is None:
            raise ValueError(
                "future_horizon_tokens is required with future root conditions"
            )
        future_horizon_tokens = torch.as_tensor(
            raw_future_horizon,
            device=future_value.device,
            dtype=torch.long,
        )
        if future_horizon_tokens.ndim == 0:
            future_horizon_tokens = future_horizon_tokens.expand(
                future_value.shape[0]
            )
        else:
            future_horizon_tokens = future_horizon_tokens.reshape(-1)

    condition = LDFCondition(
        text_context=list(text_context),
        text_null_context=list(text_null_context),
        root_condition_value=root_value,
        root_condition_mask=root_mask,
        body_condition_value=body_condition_value,
        body_condition_mask=body_condition_mask,
        future_root_condition_value=future_value,
        future_root_condition_mask=future_mask,
        future_timeline_position_ids=future_timeline_position_ids,
        future_valid_mask=future_valid,
        future_horizon_tokens=future_horizon_tokens,
    )
    latent_dim = None if body_condition_value is None else body_condition_value.shape[-1]
    validator = (
        condition.validate
        if validate_numerics
        else condition.validate_structure
    )
    validator(
        batch_size=batch_size,
        token_length=token_length,
        latent_dim=latent_dim,
    )
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


def expand_null_timeline(
    text_null_context: list[torch.Tensor], token_length: int
) -> list[torch.Tensor]:
    """Repeat each sample's null embedding across its token timeline by reference."""

    if token_length <= 0:
        raise ValueError("token_length must be positive")
    return [
        text_null_context[batch_index]
        for batch_index in range(len(text_null_context))
        for _ in range(token_length)
    ]


def create_cfg_condition(
    condition: LDFCondition, *, token_length: int
) -> dict[str, LDFCondition]:
    """Create joint/text/constraint/history branches without mutating input."""
    null_timeline = expand_null_timeline(
        condition.text_null_context, token_length
    )
    history = _without_constraints(condition)
    history = replace(history, text_context=null_timeline)
    text_only = _without_constraints(condition)
    constraint_only = replace(condition, text_context=null_timeline)
    return {
        "joint": condition,
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
    "expand_null_timeline",
    "normalize_features",
    "unnormalize_features",
]
