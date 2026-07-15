"""Frozen text-embedding lookup used by LDF training.

The training corpus is finite, so UMT5 runs once in the offline
``pretokenize_t5_text.py`` tool.  This module only validates and retrieves the
resulting CPU tensors; it does not own a text encoder or model-side condition
semantics.
"""

from __future__ import annotations

from pathlib import Path

import torch


class TextEmbeddingLookup:
    """Read a caption-to-embedding table without loading UMT5 on the train GPU."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_dim: int,
        expected_text_len: int,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"text embedding table not found at {self.path}")
        payload = torch.load(
            self.path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("embeddings"), dict
        ):
            raise ValueError("text embedding table must contain an embeddings mapping")
        if int(payload.get("text_dim", -1)) != int(expected_dim):
            raise ValueError("text embedding dimension does not match the LDF model")
        if int(payload.get("text_len", -1)) != int(expected_text_len):
            raise ValueError("text embedding length does not match the LDF model")

        embeddings: dict[str, torch.Tensor] = {}
        for raw_text, value in payload["embeddings"].items():
            text = str(raw_text).strip()
            if text in embeddings:
                raise ValueError(
                    f"duplicate normalized caption in embedding table: {text!r}"
                )
            if not torch.is_tensor(value) or value.ndim != 2:
                raise ValueError(f"embedding for {text!r} must be [L,C]")
            if value.shape[0] <= 0 or value.shape[0] > int(expected_text_len):
                raise ValueError(f"embedding length for {text!r} is invalid")
            if value.shape[1] != int(expected_dim):
                raise ValueError(f"embedding width for {text!r} is invalid")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"embedding for {text!r} must be finite floating point")
            embeddings[text] = value.detach()
        if "" not in embeddings:
            raise ValueError("text embedding table must contain the empty prompt")
        self.embeddings = embeddings
        self.text_dim = int(expected_dim)
        self.text_len = int(expected_text_len)

    def lookup(self, texts: list[str]) -> list[torch.Tensor]:
        output = []
        missing = []
        for raw_text in texts:
            text = str(raw_text).strip()
            value = self.embeddings.get(text)
            if value is None:
                missing.append(text)
            else:
                output.append(value)
        if missing:
            preview = ", ".join(repr(text) for text in dict.fromkeys(missing[:5]))
            raise KeyError(
                f"{len(missing)} prompts are missing from {self.path}; first entries: {preview}"
            )
        return output

    def __len__(self) -> int:
        return len(self.embeddings)


__all__ = ["TextEmbeddingLookup"]
