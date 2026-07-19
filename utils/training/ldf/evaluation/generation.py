"""Deterministic full-stream and rolling-window LDF validation generation."""

from __future__ import annotations

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
    normalized_motion: HybridMotion
    root_motion: torch.Tensor
    body_motion: torch.Tensor
    prompt: EvaluationPrompt
    traces: tuple[object, ...]


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
    normalized = HybridMotion(
        module.model.normalize_root(
            root.reshape(1, token_count, FRAMES_PER_TOKEN, 5)
        ),
        torch.cat(
            [chunk.committed_motion.latent_motion for chunk in chunks], dim=1
        ),
    )
    normalized.validate()
    return GeneratedSequence(
        mode=mode,
        normalized_motion=normalized,
        root_motion=root,
        body_motion=body,
        prompt=prompt,
        traces=tuple(chunk.trace for chunk in chunks),
    )


__all__ = [
    "EvaluationPrompt",
    "GENERATION_MODES",
    "GeneratedSequence",
    "compile_evaluation_prompt",
    "generate_evaluation_sequence",
]
