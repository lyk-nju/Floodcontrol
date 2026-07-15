"""Hybrid Root/Body Latent Diffusion Forcing model.

The public model is intentionally named :class:`LDF`.  Root-first/body-second
is an internal architectural fact rather than a compatibility suffix.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace
from typing import Iterator

import torch
import torch.nn as nn

from models.tools.wan_model import (
    WanLayerNorm,
    WanTransformerBlock,
    embed_text_context,
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
from utils.token_frame import FRAMES_PER_TOKEN


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
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.freq_dim = int(freq_dim)
        self.text_dim = int(text_dim)
        self.text_len = int(text_len)
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
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(text_context) == batch_size * token_length:
            # Preserve token-aligned prompt changes as a per-sample sequence of
            # prompt summaries.  All motion tokens can cross-attend this known
            # prompt timeline without introducing a trajectory-specific mask.
            summarized = []
            for batch_idx in range(batch_size):
                rows = text_context[
                    batch_idx * token_length : (batch_idx + 1) * token_length
                ]
                summarized.append(
                    torch.stack([row.float().mean(dim=0) for row in rows], dim=0)
                )
            text_context = summarized
        if len(text_context) != batch_size:
            raise ValueError(
                f"text_context must contain B or B*T entries, got {len(text_context)}"
            )
        return embed_text_context(
            self.text_projection,
            text_context,
            text_len=self.text_len,
            device=device,
        )

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
    ) -> torch.Tensor:
        batch, length = tokens.shape[:2]
        hidden = self.input_projection(tokens) + self.region_embedding(region_ids)
        time = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, beta).float()
        )
        modulation = self.time_projection(time).reshape(
            batch, length, 6, self.hidden_dim
        )
        context, context_lens = self._prepare_text(
            text_context,
            batch_size=batch,
            token_length=motion_token_length,
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
    ):
        root_patch_dim = FRAMES_PER_TOKEN * ROOT_DIM
        input_dim = root_patch_dim + latent_dim + root_patch_dim + 3
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
        # This replacement exists only in the branch-local read-only view.
        root_view = torch.where(root_mask, root_value, noisy.root_motion)
        current = torch.cat(
            [
                root_view.flatten(2),
                noisy.latent_motion,
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

        if condition.future_root_condition_value is not None:
            if bool((valid_lengths != tokens).any()):
                raise ValueError("future tokens require a fully valid current motion window")
            future_value = condition.future_root_condition_value.to(noisy.root_motion)
            future_mask = condition.future_root_condition_mask.to(
                device=noisy.root_motion.device, dtype=torch.bool
            )
            future = self.future_projection(
                torch.cat([future_value.flatten(2), future_mask.flatten(2).float()], dim=-1)
            )
            # future_projection maps directly to the stage input width; current
            # remains in that same pre-projection feature space.
            all_tokens = torch.cat([current, future], dim=1)
            future_count = future.shape[1]
            all_beta = torch.cat(
                [inputs.beta, inputs.beta.new_zeros(batch, future_count)], dim=1
            )
            future_rope_positions = inputs.timeline_to_rope(
                condition.future_timeline_position_ids.to(
                    inputs.timeline_position_ids.device
                )
            )
            all_rope_positions = torch.cat(
                [
                    inputs.rope_position_ids,
                    future_rope_positions,
                ],
                dim=1,
            )
            all_regions = torch.cat(
                [
                    region_ids,
                    torch.full(
                        (batch, future_count),
                        2,
                        device=region_ids.device,
                        dtype=torch.long,
                    ),
                ],
                dim=1,
            )
            seq_lens = valid_lengths + condition.future_valid_mask.sum(
                dim=1, dtype=torch.long
            ).to(valid_lengths.device)

        output = self._run_blocks(
            all_tokens,
            beta=all_beta,
            region_ids=all_regions,
            seq_lens=seq_lens,
            rope_position_ids=all_rope_positions,
            text_context=condition.text_context,
            motion_token_length=tokens,
        )
        return output[:, :tokens].reshape(batch, tokens, FRAMES_PER_TOKEN, ROOT_DIM)


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
        return self._run_blocks(
            stage_input,
            beta=inputs.beta,
            region_ids=region_ids,
            seq_lens=_get_valid_lengths(inputs.history_mask, inputs.generation_mask),
            rope_position_ids=inputs.rope_position_ids,
            text_context=condition.text_context,
            motion_token_length=latent.shape[1],
        )


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
        text_len: int = 512,
        num_heads: int = 8,
        root_num_layers: int = 8,
        body_num_layers: int = 8,
        chunk_size: int = 5,
        noise_steps: int = 10,
        fps: float = 20.0,
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

    def _local_root(
        self, clean_root: torch.Tensor, previous_root_frame: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        physical = unnormalize_features(clean_root, self.root_mean, self.root_std)
        local, valid = recover_local_root(
            physical.flatten(1, 2), previous_root_frame, fps=self.fps
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
        if bool((~valid.any(dim=1)).any()):
            raise ValueError("each sample needs at least one valid motion token")
        first_token = valid.to(torch.int64).argmax(dim=1)
        batch_index = torch.arange(clean_root.shape[0], device=clean_root.device)
        root_frame = clean_root[batch_index, first_token, 0]
        physical = unnormalize_features(root_frame, self.root_mean, self.root_std)
        return project_root_heading(physical)[..., 3:5]

    def forward(self, inputs: LDFInput) -> LDFPrediction:
        """Run one joint condition branch without classifier-free guidance."""
        inputs.validate()
        root_velocity = self._predict_root(inputs, inputs.condition)
        clean_root = self._recover_root(
            inputs.noisy_motion.root_motion, inputs.beta, root_velocity
        )
        local, local_valid, normalized_local = self._local_root(
            clean_root, inputs.previous_root_frame
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
        inputs.validate()
        mode = self.cfg_mode if mode is None else str(mode)
        branches = create_cfg_condition(inputs.condition)
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
            clean_root, inputs.previous_root_frame
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

    def triangular_beta(
        self,
        *,
        timeline_position_ids: torch.Tensor,
        diffusion_time: float,
    ) -> torch.Tensor:
        return torch.clamp(
            1.0
            + timeline_position_ids.float() / float(self.chunk_size)
            - float(diffusion_time),
            min=0.0,
            max=1.0,
        )

    def _create_step_input(
        self,
        motion: HybridMotion,
        *,
        beta: torch.Tensor,
        timeline_position_ids: torch.Tensor,
        commit_index: int,
        condition: LDFCondition,
        previous_root_frame: torch.Tensor | None,
    ) -> LDFInput:
        history = timeline_position_ids < int(commit_index)
        generation = ~history
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
        total_microsteps = steps + math.ceil((tokens - 1) * steps / self.chunk_size)
        for microstep in range(total_microsteps):
            time = microstep / float(steps)
            next_time = (microstep + 1) / float(steps)
            beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=time,
            )
            next_beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=next_time,
            )
            delta = beta - next_beta
            inputs = self._create_step_input(
                motion,
                beta=beta,
                timeline_position_ids=timeline_positions,
                commit_index=int((beta <= 1e-7).sum(dim=1).min().item()),
                condition=condition,
                previous_root_frame=None,
            )
            prediction = self.predict_with_cfg(inputs, mode=cfg_mode)
            motion = HybridMotion(
                motion.root_motion
                + prediction.velocity.root_motion * delta[..., None, None],
                motion.latent_motion
                + prediction.velocity.latent_motion * delta[..., None],
            )
        return motion

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

    @torch.no_grad()
    def stream_generate_step(
        self,
        state: LDFStreamState,
        condition: LDFCondition,
        *,
        cfg_mode: str | None = None,
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
        microsteps = end_step - state.current_step
        if microsteps <= 0:
            raise RuntimeError("stream scheduler did not advance")
        for offset in range(microsteps):
            microstep = state.current_step + offset
            time = microstep / float(steps)
            next_time = (microstep + 1) / float(steps)
            beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=time,
            )
            next_beta = self.triangular_beta(
                timeline_position_ids=timeline_positions,
                diffusion_time=next_time,
            )
            delta = beta - next_beta
            inputs = self._create_step_input(
                motion,
                beta=beta,
                timeline_position_ids=timeline_positions,
                commit_index=state.commit_index,
                condition=condition,
                previous_root_frame=state.previous_root_frame,
            )
            prediction = self.predict_with_cfg(inputs, mode=cfg_mode)
            motion = HybridMotion(
                motion.root_motion
                + prediction.velocity.root_motion * delta[..., None, None],
                motion.latent_motion
                + prediction.velocity.latent_motion * delta[..., None],
            )

        new_commit = state.commit_index + 1
        local_start = state.commit_index - state.window_origin
        local_end = local_start + 1
        if local_end > length:
            raise RuntimeError("stream buffer exhausted before rolling")
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
        return self._roll_window(new_state), committed

    @torch.no_grad()
    def stream_generate(
        self,
        state: LDFStreamState,
        condition_provider,
        *,
        num_chunks: int,
        cfg_mode: str | None = None,
    ) -> Iterator[HybridMotion]:
        current = state
        for _ in range(int(num_chunks)):
            condition = (
                condition_provider(current)
                if callable(condition_provider)
                else condition_provider
            )
            current, committed = self.stream_generate_step(
                current, condition, cfg_mode=cfg_mode
            )
            yield committed

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
