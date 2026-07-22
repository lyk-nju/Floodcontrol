"""Deterministic full-stream and rolling-window LDF validation generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from utils.conditions.ldf import HybridMotion
from utils.inference import (
    GuidanceConfig,
    InferenceConfig,
    InferenceSession,
    RouteEndBehavior,
    RouteReference,
)
from utils.motion_process import rotate_motion_yaw
from utils.token_frame import FRAMES_PER_TOKEN


GENERATION_MODES = frozenset({"stream", "rolling"})


@dataclass(frozen=True)
class EvaluationPrompt:
    timeline: tuple[str, ...]
    caption: str
    tokens: tuple[str, ...]
    change_frames: np.ndarray


@dataclass(frozen=True)
class GeneratedSequence:
    mode: str
    hybrid_motion: HybridMotion
    root_motion: torch.Tensor
    body_motion: torch.Tensor
    prompt: EvaluationPrompt
    traces: tuple[object, ...]


def rotate_evaluation_sample(
    sample: dict[str, object],
    *,
    frame_count: int,
    yaw_degrees: float,
) -> dict[str, object]:
    """Return a root/body-consistent global-yaw variant of one sample."""

    frames = int(frame_count)
    root = sample["root_motion"][:frames]
    body = sample["body_motion"][:frames]
    angle = torch.tensor(
        [math.radians(float(yaw_degrees))],
        device=root.device,
        dtype=root.dtype,
    )
    rotated_root, rotated_body = rotate_motion_yaw(
        root[None],
        body[None].to(device=root.device, dtype=root.dtype),
        angle,
    )
    rotated = dict(sample)
    rotated["root_motion"] = rotated_root[0]
    rotated["body_motion"] = rotated_body[0].to(body)
    return rotated


def create_evaluation_initial_noise(
    module,
    *,
    window_tokens: int,
    seed: int,
    yaw_degrees: float,
) -> HybridMotion:
    """Create paired source noise and apply the physical yaw action to root5."""

    parameter = next(module.model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    generator = torch.Generator(device=device).manual_seed(int(seed))
    root = torch.randn(
        1,
        int(window_tokens),
        FRAMES_PER_TOKEN,
        5,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    latent = torch.randn(
        1,
        int(window_tokens),
        module.model.latent_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    angle = math.radians(float(yaw_degrees))
    if abs(angle) >= 1e-12:
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotated = root.clone()
        x = root[..., 0]
        z = root[..., 2]
        rotated[..., 0] = cosine * x + sine * z
        rotated[..., 2] = -sine * x + cosine * z
        heading_cosine = root[..., 3]
        heading_sine = root[..., 4]
        rotated[..., 3] = cosine * heading_cosine - sine * heading_sine
        rotated[..., 4] = sine * heading_cosine + cosine * heading_sine
        root = rotated
    return HybridMotion(root, latent)


def compile_evaluation_prompt(
    sample: dict[str, object],
    *,
    frame_count: int,
) -> EvaluationPrompt:
    """Compile one deterministic token prompt timeline from full annotations."""

    token_count = int(frame_count) // FRAMES_PER_TOKEN
    annotations = list(sample.get("text_data", []))
    if not annotations:
        return EvaluationPrompt(
            timeline=("",) * token_count,
            caption="",
            tokens=(),
            change_frames=np.asarray([0, frame_count], dtype=np.int64),
        )

    indexed = []
    for order, annotation in enumerate(annotations):
        start = max(0, int(annotation["start_frame"]))
        end = min(frame_count, int(annotation["end_frame"]))
        if end > start:
            indexed.append((start, end, order, annotation))
    if not indexed:
        return EvaluationPrompt(
            timeline=("",) * token_count,
            caption="",
            tokens=(),
            change_frames=np.asarray([0, frame_count], dtype=np.int64),
        )

    dataset = str(sample.get("dataset", ""))
    if dataset == "HumanML3D":
        selected = max(
            indexed,
            key=lambda item: (
                int(item[0] == 0 and item[1] == frame_count),
                item[1] - item[0],
                -item[2],
            ),
        )[-1]
        text = str(selected["text"])
        return EvaluationPrompt(
            timeline=(text,) * token_count,
            caption=text,
            tokens=tuple(str(token) for token in selected.get("tokens", [])),
            change_frames=np.asarray([0, frame_count], dtype=np.int64),
        )

    timeline: list[str] = []
    selected_annotations: list[dict[str, object] | None] = []
    for token_index in range(token_count):
        token_start = token_index * FRAMES_PER_TOKEN
        token_end = token_start + FRAMES_PER_TOKEN
        candidates = []
        for start, end, order, annotation in indexed:
            overlap = max(0, min(token_end, end) - max(token_start, start))
            if overlap:
                candidates.append((overlap, -(end - start), -order, annotation))
        selected = max(candidates, key=lambda item: item[:-1])[-1] if candidates else None
        selected_annotations.append(selected)
        timeline.append("" if selected is None else str(selected["text"]))

    changes = [0]
    for index in range(1, token_count):
        if timeline[index] != timeline[index - 1]:
            changes.append(index * FRAMES_PER_TOKEN)
    changes.append(frame_count)
    first = next((value for value in selected_annotations if value is not None), None)
    return EvaluationPrompt(
        timeline=tuple(timeline),
        caption="" if first is None else str(first["text"]),
        tokens=tuple(() if first is None else first.get("tokens", [])),
        change_frames=np.asarray(changes, dtype=np.int64),
    )


def _text_encoder(module):
    def encode(texts: list[str], device: torch.device) -> list[torch.Tensor]:
        return [value.to(device=device) for value in module.text_embeddings.lookup(texts)]

    return encode


@torch.no_grad()
def generate_evaluation_sequence(
    module,
    sample: dict[str, object],
    *,
    mode: str,
    guidance_mode: str | None = None,
    seed: int,
    frame_count: int,
    dense_xz: bool,
    rolling_window_tokens: int,
    max_horizon_token: int,
    num_denoise_steps: int,
    initial_noise_yaw_degrees: float | None = None,
) -> GeneratedSequence:
    """Generate and causally decode one complete physical validation sequence."""

    mode = str(mode)
    if mode not in GENERATION_MODES:
        raise ValueError(f"unsupported evaluation generation mode {mode!r}")
    frames = int(frame_count)
    if frames <= 0 or frames % FRAMES_PER_TOKEN:
        raise ValueError("evaluation frame_count must be a positive multiple of four")
    target_root = sample["root_motion"][:frames]
    if target_root.ndim != 2 or target_root.shape[-1] != 5:
        raise ValueError("evaluation sample root_motion must be [F,5]")
    token_count = frames // FRAMES_PER_TOKEN
    prompt = compile_evaluation_prompt(sample, frame_count=frames)

    if mode == "stream":
        window_tokens = max(token_count, int(module.model.chunk_size) + 1)
        rolling = False
    else:
        window_tokens = int(rolling_window_tokens)
        rolling = True
    if window_tokens <= int(module.model.chunk_size):
        raise ValueError("evaluation window must be larger than the LDF chunk size")
    initial_noise = None
    if initial_noise_yaw_degrees is not None:
        initial_noise = create_evaluation_initial_noise(
            module,
            window_tokens=window_tokens,
            seed=seed,
            yaw_degrees=float(initial_noise_yaw_degrees),
        )

    session = InferenceSession(
        ldf=module.model,
        body_vae=module.vae,
        text_encoder=_text_encoder(module),
        config=InferenceConfig(
            window_tokens=window_tokens,
            max_horizon_token=int(max_horizon_token),
            num_denoise_steps=int(num_denoise_steps),
            rolling=rolling,
        ),
        guidance=GuidanceConfig(
            mode=(
                module.model.cfg_mode
                if guidance_mode is None
                else str(guidance_mode)
            ),
            scale_text=module.model.cfg_scale_text,
            scale_constraint=module.model.cfg_scale_constraint,
            scale_joint=module.model.cfg_scale_joint,
        ),
        seed=int(seed),
        initial_world_xz=target_root[0, [0, 2]].tolist(),
        initial_text=prompt.timeline[0],
        initial_noise=initial_noise,
    )
    for token_index in range(1, token_count):
        if prompt.timeline[token_index] != prompt.timeline[token_index - 1]:
            session.update_text(
                prompt.timeline[token_index], effective_token=token_index
            )
    if dense_xz:
        session.update_route(
            times=np.arange(frames, dtype=np.float32) / float(module.model.fps),
            points_xz=target_root[:, [0, 2]].detach().cpu().float().numpy(),
            reference=RouteReference.WORLD,
            end_behavior=RouteEndBehavior.HOLD,
            source="validation_dense_xz",
        )

    chunks = list(session.generate(token_count))
    if len(chunks) != token_count:
        raise RuntimeError("evaluation runtime committed the wrong number of tokens")
    if any(chunk.committed_motion is None for chunk in chunks):
        raise RuntimeError("evaluation runtime did not expose committed hybrid motion")
    root = torch.cat([chunk.root_motion for chunk in chunks], dim=1)[0]
    body = torch.cat(
        [chunk.body_prediction.body_motion(threshold=0.5) for chunk in chunks],
        dim=1,
    )[0]
    if tuple(root.shape[:1]) != (frames,) or tuple(body.shape[:1]) != (frames,):
        raise RuntimeError("evaluation decode did not recover exactly four frames per token")
    hybrid_motion = HybridMotion(
        root.reshape(1, token_count, FRAMES_PER_TOKEN, 5),
        torch.cat(
            [chunk.committed_motion.latent_motion for chunk in chunks], dim=1
        ),
    )
    hybrid_motion.validate()
    return GeneratedSequence(
        mode=mode,
        hybrid_motion=hybrid_motion,
        root_motion=root,
        body_motion=body,
        prompt=prompt,
        traces=tuple(chunk.trace for chunk in chunks),
    )


__all__ = [
    "create_evaluation_initial_noise",
    "EvaluationPrompt",
    "GENERATION_MODES",
    "GeneratedSequence",
    "compile_evaluation_prompt",
    "generate_evaluation_sequence",
    "rotate_evaluation_sample",
]
