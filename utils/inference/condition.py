"""Compile world-space runtime controls into one window-aligned LDF condition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from utils.conditions.ldf import LDFCondition, LDFStreamState, create_ldf_condition
from utils.inference.route import RoutePlan
from utils.inference.text import TextEmbeddingCache, TextInterval, TextTimeline
from utils.motion_process import ROOT_DIM
from utils.token_frame import FRAMES_PER_TOKEN


@dataclass(frozen=True)
class RootObservation:
    """A feature-masked physical world root observation at one frame."""

    frame_index: int
    value: np.ndarray
    feature_mask: np.ndarray

    def __post_init__(self) -> None:
        frame = int(self.frame_index)
        value = np.asarray(self.value, dtype=np.float32).reshape(-1).copy()
        mask = np.asarray(self.feature_mask, dtype=bool).reshape(-1).copy()
        if frame < 0:
            raise ValueError("frame_index must be non-negative")
        if tuple(value.shape) != (ROOT_DIM,) or tuple(mask.shape) != (ROOT_DIM,):
            raise ValueError("root observation value/mask must have shape [5]")
        if not bool(np.isfinite(value).all()):
            raise ValueError("root observation must contain only finite values")
        if bool(mask[3]) != bool(mask[4]):
            raise ValueError("heading cos/sin observations must be masked together")
        if bool(mask[3]):
            norm = float(np.linalg.norm(value[3:5]))
            if abs(norm - 1.0) > 1e-4:
                raise ValueError("observed heading cos/sin must lie on the unit circle")
        value.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "frame_index", frame)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "feature_mask", mask)


class RootObservationTimeline:
    """Feature-wise root observations indexed by absolute frame."""

    def __init__(self):
        self._observations: dict[int, RootObservation] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def observations(self) -> tuple[RootObservation, ...]:
        return tuple(self._observations[index] for index in sorted(self._observations))

    def update(self, observation: RootObservation) -> None:
        if not isinstance(observation, RootObservation):
            raise TypeError("observation must be RootObservation")
        previous = self._observations.get(observation.frame_index)
        if previous is None:
            self._observations[observation.frame_index] = observation
        else:
            value = previous.value.copy()
            mask = previous.feature_mask.copy()
            value[observation.feature_mask] = observation.value[
                observation.feature_mask
            ]
            mask |= observation.feature_mask
            self._observations[observation.frame_index] = RootObservation(
                observation.frame_index, value, mask
            )
        self._revision += 1

    def get(self, frame_index: int) -> RootObservation | None:
        return self._observations.get(int(frame_index))

    def restore(
        self,
        observations: Iterable[RootObservation],
        *,
        revision: int,
    ) -> None:
        values = tuple(observations)
        if int(revision) < 0:
            raise ValueError("observation revision must be non-negative")
        if len({item.frame_index for item in values}) != len(values):
            raise ValueError("root observation snapshot contains duplicate frames")
        self._observations = {item.frame_index: item for item in values}
        self._revision = int(revision)


@dataclass(frozen=True)
class CompiledCondition:
    """An LDF condition stamped with the state coordinates it was built for."""

    window_origin: int
    commit_index: int
    window_tokens: int
    text_revision: int
    route_revision: int
    observation_revision: int
    ldf_condition: LDFCondition

    def validate_for(self, state: LDFStreamState) -> None:
        if self.window_origin != state.window_origin:
            raise ValueError("compiled condition belongs to a different window origin")
        if self.commit_index != state.commit_index:
            raise ValueError("compiled condition belongs to a different commit index")
        if self.window_tokens != state.noisy_motion.token_length:
            raise ValueError("compiled condition has the wrong motion window length")
        self.ldf_condition.validate(
            batch_size=state.noisy_motion.batch_size,
            token_length=state.noisy_motion.token_length,
            latent_dim=state.noisy_motion.latent_motion.shape[-1],
        )


class InferenceConditionCompiler:
    """Compile one batch-1 session's authoritative world controls."""

    def __init__(
        self,
        *,
        text_embeddings: TextEmbeddingCache,
        root_mean: torch.Tensor,
        root_std: torch.Tensor,
        fps: float = 20.0,
        future_constraint_tokens: int = 0,
    ):
        mean = torch.as_tensor(root_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(root_std, dtype=torch.float32).reshape(-1)
        if tuple(mean.shape) != (ROOT_DIM,) or tuple(std.shape) != (ROOT_DIM,):
            raise ValueError("root statistics must have shape [5]")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("root statistics must be finite")
        if bool((std <= 0).any()):
            raise ValueError("root_std must be strictly positive")
        if not np.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("fps must be finite and positive")
        if int(future_constraint_tokens) < 0:
            raise ValueError("future_constraint_tokens must be non-negative")
        self.text_embeddings = text_embeddings
        self.root_mean = mean.clone()
        self.root_std = std.clone()
        self.fps = float(fps)
        self.future_constraint_tokens = int(future_constraint_tokens)

    @staticmethod
    def _origin_array(origin_xz: torch.Tensor | np.ndarray) -> np.ndarray:
        if torch.is_tensor(origin_xz):
            origin = origin_xz.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            origin = np.asarray(origin_xz, dtype=np.float32)
        origin = origin.reshape(-1)
        if tuple(origin.shape) != (2,) or not bool(np.isfinite(origin).all()):
            raise ValueError("origin_xz must be a finite [2] value")
        return origin

    def _compile_root_frames(
        self,
        frame_indices: np.ndarray,
        *,
        route: RoutePlan | None,
        observations: RootObservationTimeline,
        origin_xz: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
        clear_before_token: int | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        frame_indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        physical = np.zeros((len(frame_indices), ROOT_DIM), dtype=np.float32)
        mask = np.zeros((len(frame_indices), ROOT_DIM), dtype=bool)

        if route is not None:
            sampled, route_valid = route.sample_frames(frame_indices, fps=self.fps)
            model_xz = sampled - origin_xz[None]
            physical[route_valid, 0] = model_xz[route_valid, 0]
            physical[route_valid, 2] = model_xz[route_valid, 1]
            mask[route_valid, 0] = True
            mask[route_valid, 2] = True

        for local_index, frame_index in enumerate(frame_indices.tolist()):
            observation = observations.get(frame_index)
            if observation is None:
                continue
            value = observation.value.copy()
            value[0] -= origin_xz[0]
            value[2] -= origin_xz[1]
            selected = observation.feature_mask
            physical[local_index, selected] = value[selected]
            mask[local_index, selected] = True

        if clear_before_token is not None:
            committed = (frame_indices // FRAMES_PER_TOKEN) < int(clear_before_token)
            mask[committed] = False
        if not bool(mask.any()):
            return None, None

        physical_tensor = torch.as_tensor(physical, device=device, dtype=dtype)
        mask_tensor = torch.as_tensor(mask, device=device, dtype=torch.bool)
        normalized = (
            physical_tensor - self.root_mean.to(device=device, dtype=dtype)
        ) / self.root_std.to(device=device, dtype=dtype)
        normalized = torch.where(mask_tensor, normalized, torch.zeros_like(normalized))
        return normalized[None], mask_tensor[None]

    def compile(
        self,
        state: LDFStreamState,
        *,
        text_timeline: TextTimeline,
        route: RoutePlan | None,
        route_revision: int,
        observations: RootObservationTimeline,
        origin_xz: torch.Tensor | np.ndarray,
    ) -> CompiledCondition:
        """Build conditions for exactly the current LDF window and revision."""

        state.validate()
        if state.noisy_motion.batch_size != 1:
            raise ValueError("online InferenceConditionCompiler currently supports batch size 1")
        device = state.noisy_motion.root_motion.device
        dtype = state.noisy_motion.root_motion.dtype
        window_tokens = state.noisy_motion.token_length
        token_positions = range(
            state.window_origin, state.window_origin + window_tokens
        )
        text_context = self.text_embeddings.encode(
            text_timeline.resolve(token_positions), device=device
        )
        text_null_context = self.text_embeddings.encode([""], device=device)

        frame_start = state.window_origin * FRAMES_PER_TOKEN
        current_frames = np.arange(
            frame_start,
            frame_start + window_tokens * FRAMES_PER_TOKEN,
            dtype=np.int64,
        )
        origin = self._origin_array(origin_xz)
        root_value, root_mask = self._compile_root_frames(
            current_frames,
            route=route,
            observations=observations,
            origin_xz=origin,
            device=device,
            dtype=dtype,
            clear_before_token=state.commit_index,
        )

        future_value = future_mask = None
        future_positions = None
        if self.future_constraint_tokens:
            future_start_token = state.window_origin + window_tokens
            future_positions = torch.arange(
                future_start_token,
                future_start_token + self.future_constraint_tokens,
                device=device,
                dtype=torch.long,
            )
            future_frames = np.arange(
                future_start_token * FRAMES_PER_TOKEN,
                (future_start_token + self.future_constraint_tokens)
                * FRAMES_PER_TOKEN,
                dtype=np.int64,
            )
            future_value, future_mask = self._compile_root_frames(
                future_frames,
                route=route,
                observations=observations,
                origin_xz=origin,
                device=device,
                dtype=dtype,
                clear_before_token=None,
            )
            if future_value is None:
                future_positions = None

        condition = create_ldf_condition(
            {
                "text_context": text_context,
                "text_null_context": text_null_context,
                "root_condition_value": root_value,
                "root_condition_mask": root_mask,
                "future_root_condition_value": future_value,
                "future_root_condition_mask": future_mask,
                "future_timeline_position_ids": future_positions,
            }
        )
        compiled = CompiledCondition(
            window_origin=state.window_origin,
            commit_index=state.commit_index,
            window_tokens=window_tokens,
            text_revision=text_timeline.revision,
            route_revision=int(route_revision),
            observation_revision=observations.revision,
            ldf_condition=condition,
        )
        compiled.validate_for(state)
        return compiled


__all__ = [
    "CompiledCondition",
    "InferenceConditionCompiler",
    "RootObservation",
    "RootObservationTimeline",
    "TextInterval",
]
