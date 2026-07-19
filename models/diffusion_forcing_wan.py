"""Hybrid Root/Body Latent Diffusion Forcing model.

The public model is intentionally named :class:`LDF`.  Root-first/body-second
is an internal architectural fact rather than a compatibility suffix.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import torch
import torch.nn as nn

from models.tools.wan_model import (
    PromptQueryMap,
    WanLayerNorm,
    WanTransformerBlock,
    create_prompt_query_map,
    prepare_unique_text_context,
    sinusoidal_embedding_1d,
)
from utils.conditions.ldf import (
    HybridMotion,
    LDFCondition,
    LDFInput,
    LDFPrediction,
    LDFStreamState,
    create_cfg_condition,
    normalize_features,
    unnormalize_features,
)
from utils.motion_process import (
    LOCAL_ROOT_DIM,
    ROOT_DIM,
    project_root_heading,
    recover_local_root,
)
from utils.token_frame import FRAMES_PER_TOKEN, MOTION_FPS


def _as_stats(name: str, value, expected_dim: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != (expected_dim,):
        raise ValueError(f"{name} must have shape [{expected_dim}], got {tuple(tensor.shape)}")
    if name.endswith("std") and bool((tensor <= 0).any()):
        raise ValueError(f"{name} must be strictly positive")
    return tensor


def _prepare_condition(
    value: torch.Tensor | None,
    mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if value is None:
        return torch.zeros_like(reference), torch.zeros_like(reference, dtype=torch.bool)
    return value.to(reference), mask.to(device=reference.device, dtype=torch.bool)


def _get_valid_lengths(history_mask: torch.Tensor, generation_mask: torch.Tensor) -> torch.Tensor:
    return (history_mask | generation_mask).sum(dim=1, dtype=torch.long)


class TransformerStage(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        freq_dim: int,
        text_dim: int,
        text_len: int,
        num_heads: int,
        num_layers: int,
        time_embedding_scale: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.freq_dim = int(freq_dim)
        self.text_dim = int(text_dim)
        self.text_len = int(text_len)
        self.time_embedding_scale = float(time_embedding_scale)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.region_embedding = nn.Embedding(3, hidden_dim)
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6)
        )
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    hidden_dim,
                    ffn_dim,
                    num_heads,
                    qk_norm=True,
                    cross_attn_norm=True,
                    causal=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = WanLayerNorm(hidden_dim, elementwise_affine=True)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def _prepare_text(
        self,
        text_context: list[torch.Tensor],
        *,
        batch_size: int,
        token_length: int,
        query_token_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, PromptQueryMap]:
        if tuple(query_token_indices.shape[:1]) != (batch_size,):
            raise ValueError("query_token_indices must have shape [B,L]")
        if query_token_indices.dtype != torch.long:
            raise TypeError("query_token_indices must be int64")

        if len(text_context) != batch_size * token_length:
            raise ValueError(
                "text_context must contain exactly B*T token entries, "
                f"got {len(text_context)} for B={batch_size}, T={token_length}"
            )
        raw_context, unique_lengths, prompt_ids = prepare_unique_text_context(
            self.text_projection,
            text_context,
            text_len=self.text_len,
            device=device,
        )
        prompt_ids = prompt_ids.reshape(batch_size, token_length)

        query_length = query_token_indices.shape[1]
        valid_index = (query_token_indices >= 0) & (
            query_token_indices < token_length
        )
        valid_query = (
            torch.arange(query_length, device=device)[None] < seq_lens[:, None]
        )
        text_query_mask = valid_index & valid_query
        indices = query_token_indices.clamp(min=0, max=max(token_length - 1, 0))
        batch_indices = torch.arange(batch_size, device=device)[:, None]
        gathered_prompt_ids = prompt_ids[batch_indices, indices]
        gathered_prompt_ids = torch.where(
            text_query_mask,
            gathered_prompt_ids,
            torch.full_like(gathered_prompt_ids, -1),
        )
        prompt_map = create_prompt_query_map(
            gathered_prompt_ids,
            seq_lens,
            query_mask=text_query_mask,
            prompt_count=raw_context.shape[0],
            max_context_length=raw_context.shape[1],
        )
        if prompt_map.used_prompt_ids.numel() == 0:
            packed_raw_context = raw_context.new_empty(0, raw_context.shape[-1])
            used_lengths = unique_lengths.new_empty(0)
        else:
            used_raw_context = raw_context.index_select(
                0, prompt_map.used_prompt_ids
            )
            used_lengths = unique_lengths.index_select(
                0, prompt_map.used_prompt_ids
            ).clamp_min(1).to(dtype=torch.int32)
            valid_context = (
                torch.arange(raw_context.shape[1], device=device)[None]
                < used_lengths[:, None]
            )
            packed_raw_context = used_raw_context[valid_context]
        projected_context = self.text_projection(packed_raw_context)
        return projected_context, used_lengths, prompt_map

    def _run_blocks(
        self,
        tokens: torch.Tensor,
        *,
        beta: torch.Tensor,
        region_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        rope_position_ids: torch.Tensor,
        text_context: list[torch.Tensor],
        motion_token_length: int,
        text_query_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = tokens.shape[:2]
        if text_query_indices is None:
            text_query_indices = torch.arange(
                length, device=tokens.device, dtype=torch.long
            )[None].expand(batch, -1)
        hidden = self.input_projection(tokens) + self.region_embedding(region_ids)
        time = self.time_embedding(
            sinusoidal_embedding_1d(
                self.freq_dim, beta * self.time_embedding_scale
            ).float()
        )
        modulation = self.time_projection(time).reshape(
            batch, length, 6, self.hidden_dim
        )
        context, context_lens, text_prompt_map = self._prepare_text(
            text_context,
            batch_size=batch,
            token_length=motion_token_length,
            query_token_indices=text_query_indices,
            seq_lens=seq_lens,
            device=tokens.device,
        )
        for block in self.blocks:
            hidden = block(
                hidden,
                modulation=modulation,
                seq_lens=seq_lens,
                rope_position_ids=rope_position_ids,
                context=context,
                context_lens=context_lens,
                text_prompt_map=text_prompt_map,
            )
        return self.output_projection(self.output_norm(hidden))


class RootTransformer(TransformerStage):
    """Predict normalized explicit-root velocity from the full hybrid state."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        freq_dim: int,
        text_dim: int,
        text_len: int,
        num_heads: int,
        num_layers: int,
        time_embedding_scale: float,
    ):
        root_patch_dim = FRAMES_PER_TOKEN * ROOT_DIM
        input_dim = root_patch_dim + latent_dim + root_patch_dim * 2 + 3
        super().__init__(
            input_dim=input_dim,
            output_dim=root_patch_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            text_len=text_len,
            num_heads=num_heads,
            num_layers=num_layers,
            time_embedding_scale=time_embedding_scale,
        )
        self.root_patch_dim = root_patch_dim
        self.future_projection = nn.Linear(root_patch_dim * 2, input_dim)

    def forward(self, inputs: LDFInput, condition: LDFCondition) -> torch.Tensor:
        noisy = inputs.noisy_motion
        batch, tokens = noisy.root_motion.shape[:2]
        root_value, root_mask = _prepare_condition(
            condition.root_condition_value,
            condition.root_condition_mask,
            noisy.root_motion,
        )
        observed_root = torch.where(
            root_mask,
            root_value,
            torch.zeros_like(root_value),
        )
        current = torch.cat(
            [
                noisy.root_motion.flatten(2),
                noisy.latent_motion,
                observed_root.flatten(2),
                root_mask.flatten(2).float(),
                inputs.beta[..., None],
                inputs.history_mask[..., None].float(),
                inputs.generation_mask[..., None].float(),
            ],
            dim=-1,
        )
        valid_lengths = _get_valid_lengths(inputs.history_mask, inputs.generation_mask)
        region_ids = torch.where(
            inputs.history_mask,
            torch.zeros_like(inputs.timeline_position_ids),
            torch.ones_like(inputs.timeline_position_ids),
        )
        all_tokens = current
        all_beta = inputs.beta
        all_rope_positions = inputs.rope_position_ids
        all_regions = region_ids
        seq_lens = valid_lengths
        text_query_indices = torch.arange(
            tokens, device=current.device, dtype=torch.long
        )[None].expand(batch, -1)
        scatter_lengths = valid_lengths

        if condition.future_root_condition_value is not None:
            future_value = condition.future_root_condition_value.to(noisy.root_motion)
            future_mask = condition.future_root_condition_mask.to(
                device=noisy.root_motion.device, dtype=torch.bool
            )
            # A feature mask is an information boundary, not only metadata.
            # Values in unobserved root channels must never reach the projection.
            future_value = torch.where(
                future_mask, future_value, torch.zeros_like(future_value)
            )
            future = self.future_projection(
                torch.cat([future_value.flatten(2), future_mask.flatten(2).float()], dim=-1)
            )
            # Autocast may produce BF16 projection outputs while ``current`` and
            # the packed token buffer remain FP32. Indexed assignment requires
            # an exact dtype match, so the packed stream follows motion dtype.
            future = future.to(dtype=current.dtype)
            future_attention_mask = inputs.future_attention_mask().to(
                device=current.device
            )
            future_count = future_attention_mask.sum(
                dim=1, dtype=torch.long
            ).to(valid_lengths.device)
            future_rope_positions = inputs.timeline_to_rope(
                condition.future_timeline_position_ids.to(
                    inputs.timeline_position_ids.device
                )
            )
            seq_lens = valid_lengths + future_count
            packed_length = int(seq_lens.max().item())
            all_tokens = current.new_zeros(batch, packed_length, current.shape[-1])
            all_beta = inputs.beta.new_zeros(batch, packed_length)
            all_rope_positions = inputs.rope_position_ids.new_zeros(
                batch, packed_length
            )
            all_regions = region_ids.new_zeros(batch, packed_length)
            text_query_indices = region_ids.new_full(
                (batch, packed_length), -1
            )
            batch_grid = torch.arange(batch, device=current.device)[:, None]
            motion_positions = torch.arange(tokens, device=current.device)[None]
            motion_valid = motion_positions < valid_lengths[:, None]
            motion_batch = batch_grid.expand(-1, tokens)[motion_valid]
            motion_index = motion_positions.expand(batch, -1)[motion_valid]
            all_tokens[motion_batch, motion_index] = current[
                motion_batch, motion_index
            ]
            all_beta[motion_batch, motion_index] = inputs.beta[
                motion_batch, motion_index
            ]
            all_rope_positions[motion_batch, motion_index] = (
                inputs.rope_position_ids[motion_batch, motion_index]
            )
            all_regions[motion_batch, motion_index] = region_ids[
                motion_batch, motion_index
            ]
            text_query_indices[motion_batch, motion_index] = motion_index

            future_tokens = future.shape[1]
            future_positions = torch.arange(
                future_tokens, device=current.device
            )[None]
            future_valid = future_attention_mask
            future_batch = batch_grid.expand(-1, future_tokens)[future_valid]
            future_index = future_positions.expand(batch, -1)[future_valid]
            future_rank = future_valid.cumsum(dim=1, dtype=torch.long) - 1
            future_destination = (
                valid_lengths[:, None] + future_rank
            )[future_valid]
            all_tokens[future_batch, future_destination] = future[
                future_batch, future_index
            ]
            all_rope_positions[future_batch, future_destination] = (
                future_rope_positions[future_batch, future_index]
            )
            all_regions[future_batch, future_destination] = 2
        else:
            packed_length = int(valid_lengths.max().item())
            all_tokens = current[:, :packed_length]
            all_beta = inputs.beta[:, :packed_length]
            all_rope_positions = inputs.rope_position_ids[:, :packed_length]
            all_regions = region_ids[:, :packed_length]
            text_query_indices = text_query_indices[:, :packed_length]

        output = self._run_blocks(
            all_tokens,
            beta=all_beta,
            region_ids=all_regions,
            seq_lens=seq_lens,
            rope_position_ids=all_rope_positions,
            text_context=condition.text_context,
            motion_token_length=tokens,
            text_query_indices=text_query_indices,
        )
        root_output = output.new_zeros(batch, tokens, self.root_patch_dim)
        output_positions = torch.arange(output.shape[1], device=output.device)[None]
        motion_output_valid = output_positions < scatter_lengths[:, None]
        output_batch = torch.arange(batch, device=output.device)[:, None].expand(
            -1, output.shape[1]
        )[motion_output_valid]
        output_index = output_positions.expand(batch, -1)[motion_output_valid]
        root_output[output_batch, output_index] = output[
            output_batch, output_index
        ]
        # ``future_projection`` is data-dependent: late/short windows and
        # constraint dropout can leave one DDP rank with no future tokens while
        # peer ranks still use this layer.  Keep the parameter graph static by
        # attaching a numerically-zero dependency.  The no-future rank then
        # contributes a real zero gradient to the same reduction bucket instead
        # of leaving the parameters unused and desynchronizing DDP collectives.
        future_parameter_dependency = sum(
            parameter.sum() for parameter in self.future_projection.parameters()
        )
        root_output = root_output + (
            future_parameter_dependency.to(dtype=root_output.dtype) * 0.0
        )
        return root_output.reshape(batch, tokens, FRAMES_PER_TOKEN, ROOT_DIM)


class BodyTransformer(TransformerStage):
    """Predict body-latent velocity conditioned on one authoritative clean root."""

    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        freq_dim: int,
        text_dim: int,
        text_len: int,
        num_heads: int,
        num_layers: int,
        time_embedding_scale: float,
    ):
        local_patch_dim = FRAMES_PER_TOKEN * LOCAL_ROOT_DIM
        input_dim = latent_dim + local_patch_dim + LOCAL_ROOT_DIM * FRAMES_PER_TOKEN + 2 + 3
        super().__init__(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            text_len=text_len,
            num_heads=num_heads,
            num_layers=num_layers,
            time_embedding_scale=time_embedding_scale,
        )
        self.latent_dim = int(latent_dim)

    def forward(
        self,
        inputs: LDFInput,
        condition: LDFCondition,
        normalized_local_root: torch.Tensor,
        local_root_valid: torch.Tensor,
        body_heading_condition: torch.Tensor,
    ) -> torch.Tensor:
        latent = inputs.noisy_motion.latent_motion
        body_value, body_mask = _prepare_condition(
            condition.body_condition_value,
            condition.body_condition_mask,
            latent,
        )
        latent_view = torch.where(body_mask, body_value, latent)
        if tuple(body_heading_condition.shape) != (latent.shape[0], 2):
            raise ValueError("body_heading_condition must be [B,2]")
        stage_input = torch.cat(
            [
                latent_view,
                normalized_local_root.flatten(2),
                local_root_valid.flatten(2).float(),
                body_heading_condition[:, None].expand(-1, latent.shape[1], -1),
                inputs.beta[..., None],
                inputs.history_mask[..., None].float(),
                inputs.generation_mask[..., None].float(),
            ],
            dim=-1,
        )
        region_ids = torch.where(
            inputs.history_mask,
            torch.zeros_like(inputs.timeline_position_ids),
            torch.ones_like(inputs.timeline_position_ids),
        )
        seq_lens = _get_valid_lengths(inputs.history_mask, inputs.generation_mask)
        visible_tokens = int(seq_lens.max().item())
        output = self._run_blocks(
            stage_input[:, :visible_tokens],
            beta=inputs.beta[:, :visible_tokens],
            region_ids=region_ids[:, :visible_tokens],
            seq_lens=seq_lens,
            rope_position_ids=inputs.rope_position_ids[:, :visible_tokens],
            text_context=condition.text_context,
            motion_token_length=latent.shape[1],
            text_query_indices=torch.arange(
                visible_tokens, device=latent.device, dtype=torch.long
            )[None].expand(latent.shape[0], -1),
        )
        if visible_tokens == latent.shape[1]:
            return output
        padded = output.new_zeros(
            latent.shape[0], latent.shape[1], self.latent_dim
        )
        padded[:, :visible_tokens] = output
        return padded


class LDF(nn.Module):
    """Root-first/body-second hybrid latent diffusion forcing model."""

    def __init__(
        self,
        *,
        latent_dim: int,
        root_mean,
        root_std,
        local_root_mean,
        local_root_std,
        hidden_dim: int = 1024,
        ffn_dim: int = 2048,
        freq_dim: int = 256,
        text_dim: int = 4096,
        text_len: int = 128,
        num_heads: int = 8,
        root_num_layers: int = 8,
        body_num_layers: int = 8,
        chunk_size: int = 5,
        noise_steps: int = 10,
        fps: float = MOTION_FPS,
        time_embedding_scale: float = 1.0,
        cfg_mode: str = "separated",
        cfg_scale_text: float = 1.0,
        cfg_scale_constraint: float = 1.0,
        cfg_scale_joint: float = 1.0,
        prediction_type: str = "vel",
    ):
        super().__init__()
        if prediction_type != "vel":
            raise ValueError("new LDF supports v-predict only")
        if chunk_size <= 0 or noise_steps <= 0 or noise_steps % chunk_size:
            raise ValueError("noise_steps must be positive and divisible by chunk_size")
        if cfg_mode not in {"nocfg", "joint", "separated"}:
            raise ValueError(f"unsupported cfg_mode {cfg_mode!r}")
        self.latent_dim = int(latent_dim)
        self.chunk_size = int(chunk_size)
        self.noise_steps = int(noise_steps)
        self.fps = float(fps)
        self.time_embedding_scale = float(time_embedding_scale)
        self.cfg_mode = str(cfg_mode)
        self.cfg_scale_text = float(cfg_scale_text)
        self.cfg_scale_constraint = float(cfg_scale_constraint)
        self.cfg_scale_joint = float(cfg_scale_joint)
        self.register_buffer("root_mean", _as_stats("root_mean", root_mean, ROOT_DIM))
        self.register_buffer("root_std", _as_stats("root_std", root_std, ROOT_DIM))
        self.register_buffer(
            "local_root_mean",
            _as_stats("local_root_mean", local_root_mean, LOCAL_ROOT_DIM),
        )
        self.register_buffer(
            "local_root_std",
            _as_stats("local_root_std", local_root_std, LOCAL_ROOT_DIM),
        )
        common = dict(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            text_len=text_len,
            num_heads=num_heads,
            time_embedding_scale=time_embedding_scale,
        )
        self.root_transformer = RootTransformer(
            **common, num_layers=root_num_layers
        )
        self.body_transformer = BodyTransformer(
            **common, num_layers=body_num_layers
        )

    def _recover_root(
        self, noisy_root: torch.Tensor, beta: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        clean = noisy_root + beta[..., None, None] * velocity
        physical = unnormalize_features(clean, self.root_mean, self.root_std)
        physical = project_root_heading(physical)
        return normalize_features(physical, self.root_mean, self.root_std)

    def normalize_root(self, physical_root: torch.Tensor) -> torch.Tensor:
        """Normalize physical root5 values with this LDF's frozen statistics."""

        if not torch.is_tensor(physical_root) or physical_root.shape[-1] != ROOT_DIM:
            raise ValueError("physical_root must be a tensor ending in root5")
        if not physical_root.is_floating_point():
            raise TypeError("physical_root must be floating point")
        return normalize_features(physical_root, self.root_mean, self.root_std)

    def denormalize_root(self, normalized_root: torch.Tensor) -> torch.Tensor:
        """Restore normalized root5 values to physical model coordinates."""

        if not torch.is_tensor(normalized_root) or normalized_root.shape[-1] != ROOT_DIM:
            raise ValueError("normalized_root must be a tensor ending in root5")
        if not normalized_root.is_floating_point():
            raise TypeError("normalized_root must be floating point")
        return unnormalize_features(normalized_root, self.root_mean, self.root_std)

    def _project_normalized_root(self, normalized_root: torch.Tensor) -> torch.Tensor:
        physical = self.denormalize_root(normalized_root)
        return self.normalize_root(project_root_heading(physical))

    def _local_root(
        self,
        clean_root: torch.Tensor,
        previous_root_frame: torch.Tensor | None,
        previous_root_valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physical = unnormalize_features(clean_root, self.root_mean, self.root_std)
        local, valid = recover_local_root(
            physical.flatten(1, 2),
            previous_root_frame,
            fps=self.fps,
            previous_root_valid_mask=previous_root_valid_mask,
        )
        normalized = normalize_features(
            local, self.local_root_mean, self.local_root_std
        )
        normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
        return local, valid, normalized

    def _predict_root(self, inputs: LDFInput, condition: LDFCondition) -> torch.Tensor:
        return self.root_transformer(inputs, condition)

    def _predict_body(
        self,
        inputs: LDFInput,
        condition: LDFCondition,
        normalized_local_root: torch.Tensor,
        local_valid: torch.Tensor,
        body_heading_condition: torch.Tensor,
    ) -> torch.Tensor:
        local_for_body = normalized_local_root.detach() if self.training else normalized_local_root
        heading_for_body = (
            body_heading_condition.detach() if self.training else body_heading_condition
        )
        return self.body_transformer(
            inputs, condition, local_for_body, local_valid, heading_for_body
        )

    def _body_heading_condition(
        self, clean_root: torch.Tensor, inputs: LDFInput
    ) -> torch.Tensor:
        """Get absolute heading from the first valid clean root frame.

        The value is derived only from the authoritative Root Stage result.  It
        never reads a raw root constraint directly.
        """
        valid = inputs.history_mask | inputs.generation_mask
        first_token = valid.to(torch.int64).argmax(dim=1)
        batch_index = torch.arange(clean_root.shape[0], device=clean_root.device)
        root_frame = clean_root[batch_index, first_token, 0]
        physical = unnormalize_features(root_frame, self.root_mean, self.root_std)
        return project_root_heading(physical)[..., 3:5]

    def forward(self, inputs: LDFInput) -> LDFPrediction:
        """Run one joint condition branch without classifier-free guidance."""
        inputs.validate_structure()
        root_velocity = self._predict_root(inputs, inputs.condition)
        clean_root = self._recover_root(
            inputs.noisy_motion.root_motion, inputs.beta, root_velocity
        )
        local, local_valid, normalized_local = self._local_root(
            clean_root,
            inputs.previous_root_frame,
            inputs.previous_root_valid_mask,
        )
        heading = self._body_heading_condition(clean_root, inputs)
        latent_velocity = self._predict_body(
            inputs, inputs.condition, normalized_local, local_valid, heading
        )
        return LDFPrediction(
            velocity=HybridMotion(root_velocity, latent_velocity),
            clean_root_motion=clean_root,
            local_root_motion=local,
            local_root_feature_valid=local_valid,
        )

    @staticmethod
    def _compose_cfg(
        history: torch.Tensor,
        text: torch.Tensor,
        constraint: torch.Tensor,
        *,
        scale_text: float,
        scale_constraint: float,
    ) -> torch.Tensor:
        return history + float(scale_text) * (text - history) + float(
            scale_constraint
        ) * (constraint - history)

    def predict_with_cfg(
        self,
        inputs: LDFInput,
        *,
        mode: str | None = None,
        cfg_scale_text: float | None = None,
        cfg_scale_constraint: float | None = None,
        cfg_scale_joint: float | None = None,
    ) -> LDFPrediction:
        """Run CFG while preserving one authoritative Root-to-Body boundary."""
        inputs.validate_structure()
        mode = self.cfg_mode if mode is None else str(mode)
        branches = create_cfg_condition(
            inputs.condition, token_length=inputs.noisy_motion.token_length
        )
        scale_text = self.cfg_scale_text if cfg_scale_text is None else cfg_scale_text
        scale_constraint = (
            self.cfg_scale_constraint
            if cfg_scale_constraint is None
            else cfg_scale_constraint
        )
        scale_joint = self.cfg_scale_joint if cfg_scale_joint is None else cfg_scale_joint

        if mode == "nocfg":
            root_velocity = self._predict_root(inputs, branches["joint"])
        elif mode == "joint":
            root_history = self._predict_root(inputs, branches["history"])
            root_joint = self._predict_root(inputs, branches["joint"])
            root_velocity = root_history + float(scale_joint) * (
                root_joint - root_history
            )
        elif mode == "separated":
            root_history = self._predict_root(inputs, branches["history"])
            root_text = self._predict_root(inputs, branches["text"])
            root_constraint = self._predict_root(inputs, branches["constraint"])
            root_velocity = self._compose_cfg(
                root_history,
                root_text,
                root_constraint,
                scale_text=float(scale_text),
                scale_constraint=float(scale_constraint),
            )
        else:
            raise ValueError(f"unsupported CFG mode {mode!r}")

        clean_root = self._recover_root(
            inputs.noisy_motion.root_motion, inputs.beta, root_velocity
        )
        local, local_valid, normalized_local = self._local_root(
            clean_root,
            inputs.previous_root_frame,
            inputs.previous_root_valid_mask,
        )
        heading = self._body_heading_condition(clean_root, inputs)

        if mode == "nocfg":
            latent_velocity = self._predict_body(
                inputs, branches["joint"], normalized_local, local_valid, heading
            )
        elif mode == "joint":
            body_history = self._predict_body(
                inputs, branches["history"], normalized_local, local_valid, heading
            )
            body_joint = self._predict_body(
                inputs, branches["joint"], normalized_local, local_valid, heading
            )
            latent_velocity = body_history + float(scale_joint) * (
                body_joint - body_history
            )
        else:
            body_history = self._predict_body(
                inputs, branches["history"], normalized_local, local_valid, heading
            )
            body_text = self._predict_body(
                inputs, branches["text"], normalized_local, local_valid, heading
            )
            body_constraint = self._predict_body(
                inputs, branches["constraint"], normalized_local, local_valid, heading
            )
            latent_velocity = self._compose_cfg(
                body_history,
                body_text,
                body_constraint,
                scale_text=float(scale_text),
                scale_constraint=float(scale_constraint),
            )
        return LDFPrediction(
            velocity=HybridMotion(root_velocity, latent_velocity),
            clean_root_motion=clean_root,
            local_root_motion=local,
            local_root_feature_valid=local_valid,
        )

    def denoise_step(
        self,
        inputs: LDFInput,
        next_beta: torch.Tensor,
        *,
        use_cfg: bool,
        cfg_mode: str | None = None,
        cfg_scale_text: float | None = None,
        cfg_scale_constraint: float | None = None,
        cfg_scale_joint: float | None = None,
    ) -> tuple[HybridMotion, LDFPrediction]:
        """Advance one Euler denoise step from ``beta`` to ``next_beta``.

        This is the shared solver-state transition used by offline generation,
        streaming inference, and persistent self-forcing training.  It does not
        compile conditions, commit tokens, rebase coordinates, or roll buffers.
        """

        inputs.validate_structure()
        if not torch.is_tensor(next_beta) or tuple(next_beta.shape) != tuple(
            inputs.beta.shape
        ):
            raise ValueError("next_beta must match inputs.beta [B,T]")
        next_beta = next_beta.to(device=inputs.beta.device, dtype=inputs.beta.dtype)
        prediction = (
            self.predict_with_cfg(
                inputs,
                mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_constraint=cfg_scale_constraint,
                cfg_scale_joint=cfg_scale_joint,
            )
            if bool(use_cfg)
            else self(inputs)
        )
        delta_beta = inputs.beta - next_beta
        next_motion = HybridMotion(
            inputs.noisy_motion.root_motion
            + prediction.velocity.root_motion * delta_beta[..., None, None],
            inputs.noisy_motion.latent_motion
            + prediction.velocity.latent_motion * delta_beta[..., None],
        )
        return next_motion, prediction

    def triangular_beta(
        self,
        *,
        timeline_position_ids: torch.Tensor,
        diffusion_time: float | torch.Tensor,
    ) -> torch.Tensor:
        time = torch.as_tensor(
            diffusion_time,
            device=timeline_position_ids.device,
            dtype=torch.float32,
        )
        if time.ndim == 1:
            if time.shape[0] != timeline_position_ids.shape[0]:
                raise ValueError("per-sample diffusion_time must have shape [B]")
            time = time[:, None]
        elif time.ndim != 0:
            raise ValueError("diffusion_time must be scalar or [B]")
        return torch.clamp(
            1.0
            + timeline_position_ids.float() / float(self.chunk_size)
            - time,
            min=0.0,
            max=1.0,
        )

    def rebase_motion_state(
        self,
        motion: HybridMotion,
        beta: torch.Tensor,
        translation_xz: torch.Tensor,
    ) -> HybridMotion:
        """Change the model XZ origin without moving the Gaussian source."""

        motion.validate()
        if tuple(beta.shape) != tuple(motion.root_motion.shape[:2]):
            raise ValueError("beta must match motion [B,T]")
        translation = torch.as_tensor(
            translation_xz,
            device=motion.root_motion.device,
            dtype=motion.root_motion.dtype,
        )
        if translation.ndim == 1:
            translation = translation[None]
        if tuple(translation.shape) != (motion.batch_size, 2):
            raise ValueError("translation_xz must have shape [2] or [B,2]")
        normalized_translation = translation / self.root_std[[0, 2]].to(
            translation
        )
        root = motion.root_motion.clone()
        root[..., [0, 2]] -= (
            (1.0 - beta)[..., None, None].to(root)
            * normalized_translation[:, None, None, :].to(root)
        )
        return HybridMotion(root, motion.latent_motion.clone())

    def _create_step_input(
        self,
        motion: HybridMotion,
        *,
        beta: torch.Tensor,
        next_beta: torch.Tensor | None,
        timeline_position_ids: torch.Tensor,
        commit_index: int,
        condition: LDFCondition,
        previous_root_frame: torch.Tensor | None,
        previous_root_valid_mask: torch.Tensor | None,
    ) -> LDFInput:
        history = timeline_position_ids < int(commit_index)
        visible = beta < 1.0 - 1e-7
        if next_beta is not None:
            if tuple(next_beta.shape) != tuple(beta.shape):
                raise ValueError("next_beta must match beta [B,T]")
            visible |= next_beta < beta - 1e-7
        generation = (~history) & visible
        beta = torch.where(history, torch.zeros_like(beta), beta)
        rope_position_ids = timeline_position_ids - int(commit_index)
        return LDFInput(
            noisy_motion=motion,
            beta=beta,
            history_mask=history,
            generation_mask=generation,
            timeline_position_ids=timeline_position_ids,
            rope_position_ids=rope_position_ids,
            previous_root_frame=previous_root_frame,
            previous_root_valid_mask=previous_root_valid_mask,
            condition=condition,
        )

    @torch.no_grad()
    def generate(
        self,
        initial_noise: HybridMotion,
        condition: LDFCondition,
        *,
        num_denoise_steps: int | None = None,
        cfg_mode: str | None = None,
        cfg_scale_text: float | None = None,
        cfg_scale_constraint: float | None = None,
        cfg_scale_joint: float | None = None,
    ) -> HybridMotion:
        """Denoise a finite hybrid sequence with the triangular flow schedule."""
        initial_noise.validate()
        steps = self.noise_steps if num_denoise_steps is None else int(num_denoise_steps)
        if steps <= 0:
            raise ValueError("num_denoise_steps must be positive")
        motion = initial_noise.clone()
        batch, tokens = motion.root_motion.shape[:2]
        timeline_positions = torch.arange(
            tokens, device=motion.root_motion.device
        )[None].expand(
            batch, -1
        )
        total_denoise_steps = steps + math.ceil(
            (tokens - 1) * steps / self.chunk_size
        )
        for denoise_step_index in range(total_denoise_steps):
            time = denoise_step_index / float(steps)
            next_time = (denoise_step_index + 1) / float(steps)
            beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=time,
            )
            next_beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=next_time,
            )
            inputs = self._create_step_input(
                motion,
                beta=beta,
                next_beta=next_beta,
                timeline_position_ids=timeline_positions,
                commit_index=int((beta <= 1e-7).sum(dim=1).min().item()),
                condition=condition,
                previous_root_frame=None,
                previous_root_valid_mask=None,
            )
            motion, _ = self.denoise_step(
                inputs,
                next_beta,
                use_cfg=True,
                cfg_mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_constraint=cfg_scale_constraint,
                cfg_scale_joint=cfg_scale_joint,
            )
        return HybridMotion(
            self._project_normalized_root(motion.root_motion),
            motion.latent_motion,
        )

    def init_stream_state(
        self,
        *,
        batch_size: int,
        window_tokens: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
        initial_noise: HybridMotion | None = None,
        num_denoise_steps: int | None = None,
    ) -> LDFStreamState:
        if batch_size <= 0 or window_tokens <= self.chunk_size:
            raise ValueError("window_tokens must be larger than chunk_size")
        device = torch.device(device or self.root_mean.device)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(torch.seed())
        if initial_noise is None:
            root = torch.randn(
                batch_size,
                window_tokens,
                FRAMES_PER_TOKEN,
                ROOT_DIM,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            latent = torch.randn(
                batch_size,
                window_tokens,
                self.latent_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            initial_noise = HybridMotion(root, latent)
        else:
            initial_noise.validate()
            if initial_noise.batch_size != batch_size or initial_noise.token_length != window_tokens:
                raise ValueError("initial_noise does not match requested stream shape")
            initial_noise = initial_noise.clone()
        state = LDFStreamState(
            noisy_motion=initial_noise,
            current_step=0,
            commit_index=0,
            window_origin=0,
            epoch=0,
            previous_root_frame=None,
            rng_state=generator.get_state().clone(),
            num_denoise_steps=(
                self.noise_steps
                if num_denoise_steps is None
                else int(num_denoise_steps)
            ),
        )
        state.validate()
        return state

    def _roll_window(self, state: LDFStreamState) -> LDFStreamState:
        length = state.noisy_motion.token_length
        local_commit = state.commit_index - state.window_origin
        if local_commit <= length - self.chunk_size:
            return state
        roll = self.chunk_size
        root = state.noisy_motion.root_motion
        latent = state.noisy_motion.latent_motion
        boundary_normalized = root[:, roll - 1, -1]
        boundary_physical = unnormalize_features(
            boundary_normalized, self.root_mean, self.root_std
        )
        generator = torch.Generator(device=root.device)
        generator.set_state(state.rng_state.to(device="cpu"))
        new_root = torch.randn(
            root.shape[0],
            roll,
            FRAMES_PER_TOKEN,
            ROOT_DIM,
            device=root.device,
            dtype=root.dtype,
            generator=generator,
        )
        new_latent = torch.randn(
            latent.shape[0],
            roll,
            latent.shape[-1],
            device=latent.device,
            dtype=latent.dtype,
            generator=generator,
        )
        return replace(
            state,
            noisy_motion=HybridMotion(
                torch.cat([root[:, roll:], new_root], dim=1),
                torch.cat([latent[:, roll:], new_latent], dim=1),
            ),
            window_origin=state.window_origin + roll,
            epoch=state.epoch + 1,
            previous_root_frame=boundary_physical,
            rng_state=generator.get_state().clone(),
        )

    def rebase_stream_state(
        self,
        state: LDFStreamState,
        translation_xz: torch.Tensor,
    ) -> LDFStreamState:
        """Translate one noisy root window while preserving its flow state.

        ``translation_xz`` is a physical model-space offset which becomes part
        of the runtime world origin. Clean tokens move by the full normalized
        offset, partially noisy tokens by ``1-beta``, and pure noise is left
        unchanged. Body latent and RNG state are invariant to XZ translation.
        """

        state.validate()
        translation = torch.as_tensor(
            translation_xz,
            device=state.noisy_motion.root_motion.device,
            dtype=state.noisy_motion.root_motion.dtype,
        )
        if translation.ndim == 1:
            translation = translation[None]
        batch = state.noisy_motion.batch_size
        if tuple(translation.shape) != (batch, 2):
            raise ValueError("translation_xz must have shape [2] or [B,2]")
        if not bool(torch.isfinite(translation).all()):
            raise ValueError("translation_xz must contain only finite values")

        root = state.noisy_motion.root_motion
        positions = torch.arange(
            state.window_origin,
            state.window_origin + state.noisy_motion.token_length,
            device=root.device,
            dtype=torch.long,
        )[None].expand(batch, -1)
        beta = self.triangular_beta(
            timeline_position_ids=positions,
            diffusion_time=state.current_step / float(state.num_denoise_steps),
        ).to(dtype=root.dtype)
        motion = self.rebase_motion_state(state.noisy_motion, beta, translation)

        previous = state.previous_root_frame
        if previous is not None:
            previous = previous.clone()
            previous[..., [0, 2]] -= translation.to(previous)
        rebased = replace(
            state,
            noisy_motion=motion,
            previous_root_frame=previous,
        )
        rebased.validate()
        return rebased

    @torch.no_grad()
    def stream_generate_step(
        self,
        state: LDFStreamState,
        condition: LDFCondition,
        *,
        roll_window: bool = True,
        cfg_mode: str | None = None,
        cfg_scale_text: float | None = None,
        cfg_scale_constraint: float | None = None,
        cfg_scale_joint: float | None = None,
    ) -> tuple[LDFStreamState, HybridMotion]:
        """Advance until one token is clean, commit it, then roll if needed."""
        state.validate()
        steps = int(state.num_denoise_steps)
        motion = state.noisy_motion.clone()
        batch, length = motion.root_motion.shape[:2]
        timeline_positions = torch.arange(
            state.window_origin,
            state.window_origin + length,
            device=motion.root_motion.device,
            dtype=torch.long,
        )[None].expand(batch, -1)
        end_step = math.ceil(
            (state.commit_index + self.chunk_size)
            * steps
            / float(self.chunk_size)
        )
        denoise_steps = end_step - state.current_step
        if denoise_steps <= 0:
            raise RuntimeError("stream scheduler did not advance")
        for offset in range(denoise_steps):
            denoise_step_index = state.current_step + offset
            time = denoise_step_index / float(steps)
            next_time = (denoise_step_index + 1) / float(steps)
            beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=time,
            )
            next_beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=next_time,
            )
            inputs = self._create_step_input(
                motion,
                beta=beta,
                next_beta=next_beta,
                timeline_position_ids=timeline_positions,
                commit_index=state.commit_index,
                condition=condition,
                previous_root_frame=state.previous_root_frame,
                previous_root_valid_mask=(
                    None
                    if state.previous_root_frame is None
                    else torch.ones(
                        batch, device=motion.root_motion.device, dtype=torch.bool
                    )
                ),
            )
            motion, _ = self.denoise_step(
                inputs,
                next_beta,
                use_cfg=True,
                cfg_mode=cfg_mode,
                cfg_scale_text=cfg_scale_text,
                cfg_scale_constraint=cfg_scale_constraint,
                cfg_scale_joint=cfg_scale_joint,
            )

        new_commit = state.commit_index + 1
        local_start = state.commit_index - state.window_origin
        local_end = local_start + 1
        if local_end > length:
            raise RuntimeError("stream buffer exhausted before rolling")
        projected_root = motion.root_motion.clone()
        projected_root[:, local_start:local_end] = self._project_normalized_root(
            projected_root[:, local_start:local_end]
        )
        motion = HybridMotion(projected_root, motion.latent_motion)
        committed = HybridMotion(
            motion.root_motion[:, local_start:local_end].clone(),
            motion.latent_motion[:, local_start:local_end].clone(),
        )
        new_state = replace(
            state,
            noisy_motion=motion,
            current_step=end_step,
            commit_index=new_commit,
        )
        committed_physical = self.denormalize_root(committed.root_motion)
        translation_xz = committed_physical[:, 0, -1, [0, 2]]
        new_state = self.rebase_stream_state(new_state, translation_xz)
        return (
            self._roll_window(new_state) if bool(roll_window) else new_state
        ), committed

    @staticmethod
    def create_stream_snapshot(state: LDFStreamState) -> dict:
        state.validate()
        return {
            "noisy_root_motion": state.noisy_motion.root_motion.detach().clone(),
            "noisy_latent_motion": state.noisy_motion.latent_motion.detach().clone(),
            "current_step": int(state.current_step),
            "commit_index": int(state.commit_index),
            "window_origin": int(state.window_origin),
            "epoch": int(state.epoch),
            "previous_root_frame": (
                None
                if state.previous_root_frame is None
                else state.previous_root_frame.detach().clone()
            ),
            "rng_state": state.rng_state.detach().clone(),
            "num_denoise_steps": int(state.num_denoise_steps),
        }

    @staticmethod
    def create_stream_state_from_snapshot(snapshot: dict) -> LDFStreamState:
        required = {
            "noisy_root_motion",
            "noisy_latent_motion",
            "current_step",
            "commit_index",
            "window_origin",
            "epoch",
            "previous_root_frame",
            "rng_state",
            "num_denoise_steps",
        }
        missing = required - set(snapshot)
        if missing:
            raise ValueError(f"stream snapshot is missing fields: {sorted(missing)}")
        state = LDFStreamState(
            noisy_motion=HybridMotion(
                snapshot["noisy_root_motion"].clone(),
                snapshot["noisy_latent_motion"].clone(),
            ),
            current_step=int(snapshot["current_step"]),
            commit_index=int(snapshot["commit_index"]),
            window_origin=int(snapshot["window_origin"]),
            epoch=int(snapshot["epoch"]),
            previous_root_frame=(
                None
                if snapshot["previous_root_frame"] is None
                else snapshot["previous_root_frame"].clone()
            ),
            rng_state=snapshot["rng_state"].clone(),
            num_denoise_steps=int(snapshot["num_denoise_steps"]),
        )
        state.validate()
        return state


__all__ = ["BodyTransformer", "LDF", "RootTransformer"]
