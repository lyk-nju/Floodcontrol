"""Atomic end-to-end streaming inference over one interactive motion session."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import torch

from models.diffusion_forcing_wan import LDF
from models.vae_wan_1d import BodyVAE
from utils.conditions.ldf import HybridMotion, LDFStreamState
from utils.conditions.vae import BodyPrediction, VAEDecoderState
from utils.inference.condition import (
    CompiledCondition,
    InferenceConditionCompiler,
    RootObservation,
    RootObservationTimeline,
)
from utils.inference.route import RouteEndBehavior, RoutePlan, RouteReference
from utils.inference.text import TextEmbeddingCache, TextInterval, TextTimeline
from utils.motion_process import project_root_heading, recover_local_root
from utils.token_frame import FRAMES_PER_TOKEN


@dataclass(frozen=True)
class GuidanceConfig:
    """Per-session classifier-free guidance without shared-model mutation."""

    mode: str = "separated"
    scale_text: float = 1.0
    scale_constraint: float = 1.0
    scale_joint: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"nocfg", "joint", "separated"}:
            raise ValueError(f"unsupported CFG mode {self.mode!r}")
        for name in ("scale_text", "scale_constraint", "scale_joint"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class InferenceConfig:
    """Session scheduler and condition-window configuration."""

    window_tokens: int
    max_horizon_token: int = 0
    num_denoise_steps: int | None = None
    rolling: bool = True

    def __post_init__(self) -> None:
        if int(self.window_tokens) <= 0:
            raise ValueError("window_tokens must be positive")
        if int(self.max_horizon_token) < 0:
            raise ValueError("max_horizon_token must be non-negative")
        if self.num_denoise_steps is not None and int(self.num_denoise_steps) <= 0:
            raise ValueError("num_denoise_steps must be positive")
        object.__setattr__(self, "window_tokens", int(self.window_tokens))
        object.__setattr__(
            self, "max_horizon_token", int(self.max_horizon_token)
        )
        object.__setattr__(self, "rolling", bool(self.rolling))
        if self.num_denoise_steps is not None:
            object.__setattr__(
                self, "num_denoise_steps", int(self.num_denoise_steps)
            )


@dataclass(frozen=True)
class InferenceStepTrace:
    """Small audit record for one committed token."""

    token_index: int
    window_origin_before: int
    window_origin_after: int
    window_epoch_before: int
    window_epoch_after: int
    text_revision: int
    route_revision: int
    observation_revision: int
    rebased: bool


@dataclass(frozen=True)
class GeneratedMotionChunk:
    """One committed token expressed as four physical world frames."""

    token_index: int
    root_motion: torch.Tensor
    body_prediction: BodyPrediction
    trace: InferenceStepTrace
    committed_motion: HybridMotion | None = None

    def validate(self) -> None:
        if self.token_index < 0:
            raise ValueError("token_index must be non-negative")
        if self.root_motion.ndim != 3 or tuple(self.root_motion.shape[1:]) != (
            FRAMES_PER_TOKEN,
            5,
        ):
            raise ValueError("root_motion must be physical world [B,4,5]")
        if not bool(torch.isfinite(self.root_motion).all()):
            raise ValueError("root_motion contains non-finite values")
        self.body_prediction.validate()
        if self.body_prediction.continuous_body.shape[:2] != self.root_motion.shape[:2]:
            raise ValueError("root and body output frames must align")
        if self.committed_motion is not None:
            self.committed_motion.validate()
            if self.committed_motion.batch_size != self.root_motion.shape[0]:
                raise ValueError("committed motion and physical output batch differ")
            if self.committed_motion.token_length != 1:
                raise ValueError("one generated chunk must contain one hybrid token")


@dataclass(frozen=True)
class InferenceSnapshot:
    """In-memory deterministic snapshot of all persistent session state."""

    ldf_snapshot: dict
    decoder_state: VAEDecoderState
    origin_xz: torch.Tensor
    previous_root_frame_world: torch.Tensor | None
    text_intervals: tuple[TextInterval, ...]
    text_revision: int
    route: RoutePlan | None
    route_revision: int
    observations: tuple[RootObservation, ...]
    observation_revision: int
    guidance: GuidanceConfig


class InferenceSession:
    """Batch-1 streaming transaction over shared immutable LDF/VAE weights.

    The object is intentionally not thread-safe. A Web adapter must serialize
    updates and generation for each session while allowing different sessions
    to share the same evaluation-mode model instances.
    """

    def __init__(
        self,
        *,
        ldf: LDF,
        body_vae: BodyVAE,
        text_encoder: Callable[[list[str], torch.device], list[torch.Tensor]],
        config: InferenceConfig,
        guidance: GuidanceConfig | None = None,
        seed: int = 0,
        initial_world_xz=(0.0, 0.0),
        initial_yaw: float | None = None,
        initial_text: str = "",
        initial_noise: HybridMotion | None = None,
    ):
        if not isinstance(ldf, LDF) or not isinstance(body_vae, BodyVAE):
            raise TypeError("InferenceSession requires LDF and BodyVAE instances")
        if ldf.training or body_vae.training:
            raise ValueError("inference models must be in eval mode")
        if ldf.latent_dim != body_vae.latent_dim:
            raise ValueError("LDF and BodyVAE latent dimensions do not match")
        if abs(float(ldf.fps) - float(body_vae.fps)) > 1e-6:
            raise ValueError("LDF and BodyVAE FPS must match")
        if not isinstance(config, InferenceConfig):
            raise TypeError("config must be InferenceConfig")

        self.ldf = ldf
        self.body_vae = body_vae
        self.config = config
        self.guidance = guidance or GuidanceConfig(
            mode=ldf.cfg_mode,
            scale_text=ldf.cfg_scale_text,
            scale_constraint=ldf.cfg_scale_constraint,
            scale_joint=ldf.cfg_scale_joint,
        )
        ldf_parameter = next(ldf.parameters())
        device = ldf_parameter.device
        dtype = ldf_parameter.dtype
        vae_parameter = next(body_vae.parameters())
        if vae_parameter.device != device:
            raise ValueError("LDF and BodyVAE must be on the same device")
        vae_dtype = vae_parameter.dtype
        self._vae_device = vae_parameter.device
        self._vae_dtype = vae_dtype
        if initial_noise is not None:
            initial_noise = HybridMotion(
                initial_noise.root_motion.to(device=device, dtype=dtype),
                initial_noise.latent_motion.to(device=device, dtype=dtype),
            )
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        self.ldf_state = ldf.init_stream_state(
            batch_size=1,
            window_tokens=config.window_tokens,
            device=device,
            dtype=dtype,
            generator=generator,
            initial_noise=initial_noise,
            num_denoise_steps=config.num_denoise_steps,
        )
        self.decoder_state = body_vae.init_decoder_state(
            1, device=device, dtype=vae_dtype
        )
        origin = torch.as_tensor(initial_world_xz, device=device, dtype=dtype).reshape(-1)
        if tuple(origin.shape) != (2,) or not bool(torch.isfinite(origin).all()):
            raise ValueError("initial_world_xz must be a finite [2] value")
        self.origin_xz = origin[None].clone()
        self.previous_root_frame_world: torch.Tensor | None = None

        self.text_timeline = TextTimeline(initial_text)
        self.route: RoutePlan | None = None
        self.route_revision = 0
        self.root_observations = RootObservationTimeline()
        initial_root = np.asarray(
            [float(origin[0].item()), 0.0, float(origin[1].item()), 1.0, 0.0],
            dtype=np.float32,
        )
        initial_mask = np.asarray([True, False, True, False, False])
        if initial_yaw is not None:
            yaw = float(initial_yaw)
            if not np.isfinite(yaw):
                raise ValueError("initial_yaw must be finite")
            initial_root[3:5] = [np.cos(yaw), np.sin(yaw)]
            initial_mask[3:5] = True
        self.root_observations.update(
            RootObservation(0, initial_root, initial_mask)
        )
        self.condition_compiler = InferenceConditionCompiler(
            text_embeddings=TextEmbeddingCache(text_encoder),
            fps=ldf.fps,
            active_tokens=ldf.chunk_size,
            max_horizon_token=config.max_horizon_token,
        )

    @property
    def commit_index(self) -> int:
        return self.ldf_state.commit_index

    def update_guidance(self, guidance: GuidanceConfig) -> None:
        if not isinstance(guidance, GuidanceConfig):
            raise TypeError("guidance must be GuidanceConfig")
        self.guidance = guidance

    def update_text(self, text: str, *, effective_token: int | None = None) -> None:
        token = self.commit_index if effective_token is None else int(effective_token)
        if token < self.commit_index:
            raise ValueError("text updates cannot rewrite committed tokens")
        self.text_timeline.update(text, start_token=token)

    def update_route(
        self,
        *,
        times,
        points_xz,
        reference: RouteReference | str = RouteReference.WORLD,
        end_behavior: RouteEndBehavior | str = RouteEndBehavior.HOLD,
        source: str = "manual",
    ) -> RoutePlan:
        token = self.commit_index
        self.route_revision += 1
        route = RoutePlan(
            times=np.asarray(times, dtype=np.float32),
            points_xz=np.asarray(points_xz, dtype=np.float32),
            start_token=token,
            end_behavior=RouteEndBehavior(end_behavior),
            version=self.route_revision,
            source=source,
        )
        actor_xz = (
            self.origin_xz[0]
            if self.previous_root_frame_world is None
            else self.previous_root_frame_world[0, [0, 2]]
        )
        self.route = route.resolve_world(
            reference,
            actor_xz.detach().to(device="cpu", dtype=torch.float32).numpy(),
        )
        return self.route

    def clear_route(self) -> None:
        self.route = None
        self.route_revision += 1

    def update_root_observation(self, observation: RootObservation) -> None:
        if observation.frame_index // FRAMES_PER_TOKEN < self.commit_index:
            raise ValueError("root observations cannot rewrite committed tokens")
        self.root_observations.update(observation)

    def compile_condition(self) -> CompiledCondition:
        return self.condition_compiler.compile(
            self.ldf_state,
            text_timeline=self.text_timeline,
            route=self.route,
            route_revision=self.route_revision,
            observations=self.root_observations,
            origin_xz=self.origin_xz[0],
        )

    @torch.no_grad()
    def generate_step(self) -> GeneratedMotionChunk:
        """Generate, decode, validate, and atomically commit one token."""

        old_state = self.ldf_state
        old_epoch = old_state.epoch
        token_index = old_state.commit_index
        compiled = self.compile_condition()
        compiled.validate_for(old_state)
        candidate_ldf, committed = self.ldf.stream_generate_step(
            old_state,
            compiled.ldf_condition,
            roll_window=self.config.rolling,
            cfg_mode=self.guidance.mode,
            cfg_scale_text=self.guidance.scale_text,
            cfg_scale_constraint=self.guidance.scale_constraint,
            cfg_scale_joint=self.guidance.scale_joint,
        )
        if candidate_ldf.commit_index != token_index + 1:
            raise RuntimeError("LDF must commit exactly one token per inference step")

        world_root = committed.root_motion.clone()
        world_root[..., 0] += self.origin_xz[:, None, None, 0]
        world_root[..., 2] += self.origin_xz[:, None, None, 1]
        if not bool(torch.isfinite(world_root).all()):
            raise ValueError("committed root contains non-finite values")
        projected = project_root_heading(world_root)
        if not torch.allclose(world_root[..., 3:5], projected[..., 3:5], atol=1e-3):
            raise ValueError("committed root heading is not on the unit circle")

        local_root, local_valid = recover_local_root(
            world_root.flatten(1, 2),
            self.previous_root_frame_world,
            fps=self.ldf.fps,
        )
        candidate_decoder, body = self.body_vae.detokenize_step(
            committed.latent_motion.to(
                device=self._vae_device,
                dtype=self._vae_dtype,
            ),
            local_root.to(
                device=self._vae_device,
                dtype=self._vae_dtype,
            ),
            local_valid.to(device=self._vae_device),
            self.decoder_state,
        )
        body.validate()
        if not bool(torch.isfinite(body.continuous_body).all()) or not bool(
            torch.isfinite(body.contact_logits).all()
        ):
            raise ValueError("decoded body contains non-finite values")

        # ``stream_generate_step`` has already rebased its persistent model
        # state by this committed token's final model-space XZ.  World output
        # must use the old origin; only the next transaction sees the update.
        translation = committed.root_motion[:, 0, -1, [0, 2]].clone()
        candidate_origin = self.origin_xz + translation.to(self.origin_xz)
        rebased = True
        if candidate_ldf.epoch != old_epoch:
            if candidate_ldf.epoch != old_epoch + 1:
                raise RuntimeError("one inference step may roll the LDF window at most once")

        trace = InferenceStepTrace(
            token_index=token_index,
            window_origin_before=old_state.window_origin,
            window_origin_after=candidate_ldf.window_origin,
            window_epoch_before=old_epoch,
            window_epoch_after=candidate_ldf.epoch,
            text_revision=compiled.text_revision,
            route_revision=compiled.route_revision,
            observation_revision=compiled.observation_revision,
            rebased=rebased,
        )
        chunk = GeneratedMotionChunk(
            token_index=token_index,
            root_motion=world_root[:, 0].clone(),
            body_prediction=body,
            trace=trace,
            committed_motion=committed.clone(detach=True),
        )
        chunk.validate()

        # The only persistent mutation in the transaction happens after every
        # candidate result and coordinate update has passed validation.
        self.ldf_state = candidate_ldf
        self.decoder_state = candidate_decoder
        self.origin_xz = candidate_origin
        self.previous_root_frame_world = world_root[:, -1, -1].clone()
        return chunk

    def generate(self, num_tokens: int) -> Iterator[GeneratedMotionChunk]:
        count = int(num_tokens)
        if count < 0:
            raise ValueError("num_tokens must be non-negative")
        for _ in range(count):
            yield self.generate_step()

    def create_snapshot(self) -> InferenceSnapshot:
        return InferenceSnapshot(
            ldf_snapshot=self.ldf.create_stream_snapshot(self.ldf_state),
            decoder_state=self.decoder_state.clone(),
            origin_xz=self.origin_xz.clone(),
            previous_root_frame_world=(
                None
                if self.previous_root_frame_world is None
                else self.previous_root_frame_world.clone()
            ),
            text_intervals=self.text_timeline.intervals,
            text_revision=self.text_timeline.revision,
            route=self.route,
            route_revision=self.route_revision,
            observations=self.root_observations.observations,
            observation_revision=self.root_observations.revision,
            guidance=self.guidance,
        )

    def restore_snapshot(self, snapshot: InferenceSnapshot) -> None:
        if not isinstance(snapshot, InferenceSnapshot):
            raise TypeError("snapshot must be InferenceSnapshot")
        candidate_ldf = self.ldf.create_stream_state_from_snapshot(
            snapshot.ldf_snapshot
        )
        if candidate_ldf.noisy_motion.batch_size != 1:
            raise ValueError("inference snapshots must have batch size one")
        if candidate_ldf.noisy_motion.token_length != self.config.window_tokens:
            raise ValueError("snapshot window length does not match this session")
        origin = snapshot.origin_xz.to(
            device=self.origin_xz.device, dtype=self.origin_xz.dtype
        )
        if tuple(origin.shape) != (1, 2) or not bool(torch.isfinite(origin).all()):
            raise ValueError("snapshot origin_xz must be finite [1,2]")
        previous = snapshot.previous_root_frame_world
        if previous is not None:
            previous = previous.to(
                device=self.origin_xz.device, dtype=self.origin_xz.dtype
            )
            if tuple(previous.shape) != (1, 5) or not bool(torch.isfinite(previous).all()):
                raise ValueError("snapshot previous root must be finite [1,5]")

        text = TextTimeline("")
        text.restore(snapshot.text_intervals, revision=snapshot.text_revision)
        observations = RootObservationTimeline()
        observations.restore(
            snapshot.observations, revision=snapshot.observation_revision
        )
        self.ldf_state = candidate_ldf
        self.decoder_state = snapshot.decoder_state.clone()
        self.origin_xz = origin.clone()
        self.previous_root_frame_world = None if previous is None else previous.clone()
        self.text_timeline = text
        self.route = snapshot.route
        self.route_revision = int(snapshot.route_revision)
        self.root_observations = observations
        self.guidance = snapshot.guidance


__all__ = [
    "GeneratedMotionChunk",
    "GuidanceConfig",
    "InferenceConfig",
    "InferenceSession",
    "InferenceSnapshot",
    "InferenceStepTrace",
]
