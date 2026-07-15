"""Commit-indexed text timeline and reusable text embedding cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class TextInterval:
    """A text prompt active on the half-open token interval ``[start,end)``."""

    text: str
    start_token: int
    end_token: int | None = None

    def __post_init__(self) -> None:
        start = int(self.start_token)
        end = None if self.end_token is None else int(self.end_token)
        if start < 0:
            raise ValueError("start_token must be non-negative")
        if end is not None and end <= start:
            raise ValueError("end_token must be greater than start_token")
        object.__setattr__(self, "text", str(self.text))
        object.__setattr__(self, "start_token", start)
        object.__setattr__(self, "end_token", end)


class TextTimeline:
    """A validated prompt timeline keyed by absolute token positions."""

    def __init__(self, initial_text: str = ""):
        self._starts: dict[int, str] = {0: str(initial_text)}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def intervals(self) -> tuple[TextInterval, ...]:
        starts = sorted(self._starts)
        return tuple(
            TextInterval(
                text=self._starts[start],
                start_token=start,
                end_token=None if index + 1 == len(starts) else starts[index + 1],
            )
            for index, start in enumerate(starts)
        )

    def update(self, text: str, *, start_token: int) -> None:
        start = int(start_token)
        if start < 0:
            raise ValueError("start_token must be non-negative")
        self._starts[start] = str(text)
        self._revision += 1

    def text_at(self, token_index: int) -> str:
        token = int(token_index)
        if token < 0:
            raise ValueError("token_index must be non-negative")
        start = max(value for value in self._starts if value <= token)
        return self._starts[start]

    def resolve(self, token_indices: Iterable[int]) -> list[str]:
        return [self.text_at(int(token)) for token in token_indices]

    def restore(self, intervals: Iterable[TextInterval], *, revision: int) -> None:
        values = tuple(intervals)
        if not values or values[0].start_token != 0:
            raise ValueError("a text timeline snapshot must begin at token zero")
        if values[-1].end_token is not None:
            raise ValueError("the final text interval must be open-ended")
        if int(revision) < 0:
            raise ValueError("text revision must be non-negative")
        previous_end = 0
        starts: dict[int, str] = {}
        for index, interval in enumerate(values):
            if interval.start_token != previous_end:
                raise ValueError("text intervals must be contiguous and ordered")
            if index + 1 < len(values) and interval.end_token is None:
                raise ValueError("only the final text interval may be open-ended")
            starts[interval.start_token] = interval.text
            previous_end = (
                interval.end_token if interval.end_token is not None else previous_end
            )
        self._starts = starts
        self._revision = int(revision)


class TextEmbeddingCache:
    """Cache text encoder outputs by exact prompt content on CPU."""

    def __init__(
        self,
        encode: Callable[[list[str], torch.device], list[torch.Tensor]],
    ):
        self._encode = encode
        self._cache: dict[str, torch.Tensor] = {}

    def encode(self, texts: Iterable[str], *, device: torch.device) -> list[torch.Tensor]:
        values = [str(text) for text in texts]
        missing = list(dict.fromkeys(text for text in values if text not in self._cache))
        if missing:
            encoded = self._encode(missing, device)
            if len(encoded) != len(missing):
                raise ValueError("text encoder returned the wrong number of contexts")
            for text, context in zip(missing, encoded):
                if not torch.is_tensor(context) or context.ndim != 2:
                    raise ValueError("each text context must be a rank-2 tensor")
                if not bool(torch.isfinite(context).all()):
                    raise ValueError("text context contains non-finite values")
                self._cache[text] = context.detach().to(device="cpu").clone()
        device_values = {
            text: self._cache[text].to(device=device)
            for text in dict.fromkeys(values)
        }
        return [device_values[text] for text in values]

    def clear(self) -> None:
        self._cache.clear()


__all__ = ["TextEmbeddingCache", "TextInterval", "TextTimeline"]
