"""Frozen text-embedding lookup used by LDF training.

The training corpus is finite, so UMT5 runs once in the offline
``pretokenize_t5_text.py`` tool.  This module only validates and retrieves the
resulting CPU tensors; it does not own a text encoder or model-side condition
semantics.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch


def create_text_embedding_content_id(
    embeddings: Mapping[str, torch.Tensor],
    *,
    text_dim: int,
    text_len: int,
    checkpoint_path: str | None = None,
    tokenizer_path: str | None = None,
) -> str:
    """Create the table identity once, while the offline artifact is written."""

    digest = hashlib.blake2b(digest_size=20)

    def update_text(value: object) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    for value in (text_dim, text_len, checkpoint_path or "", tokenizer_path or ""):
        update_text(value)

    normalized: dict[str, torch.Tensor] = {}
    for raw_text, value in embeddings.items():
        text = str(raw_text).strip()
        if text in normalized:
            raise ValueError(f"duplicate normalized caption in embedding table: {text!r}")
        if not torch.is_tensor(value) or value.ndim != 2:
            raise ValueError(f"embedding for {text!r} must be [L,C]")
        if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"embedding for {text!r} must be finite floating point")
        normalized[text] = value

    for text in sorted(normalized):
        value = normalized[text].detach().cpu().contiguous()
        update_text(text)
        update_text(value.dtype)
        digest.update(int(value.shape[0]).to_bytes(8, "little"))
        digest.update(int(value.shape[1]).to_bytes(8, "little"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        content_id = payload.get("content_id")
        if not isinstance(content_id, str) or not content_id:
            raise ValueError(
                "text embedding table has no content_id; regenerate it with "
                "tools/pretokenize_t5_text.py"
            )

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
            if not value.is_floating_point():
                raise ValueError(f"embedding for {text!r} must be floating point")
            embeddings[text] = value.detach()
        if "" not in embeddings:
            raise ValueError("text embedding table must contain the empty prompt")
        self.embeddings = embeddings
        self.text_dim = int(expected_dim)
        self.text_len = int(expected_text_len)
        self.content_id = content_id
        self._finite_checked: set[str] = set()

    def lookup(self, texts: list[str]) -> list[torch.Tensor]:
        output = []
        missing = []
        for raw_text in texts:
            text = str(raw_text).strip()
            value = self.embeddings.get(text)
            if value is None:
                missing.append(text)
            else:
                if text not in self._finite_checked:
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError(
                            f"embedding for {text!r} must be finite floating point"
                        )
                    self._finite_checked.add(text)
                output.append(value)
        if missing:
            preview = ", ".join(repr(text) for text in dict.fromkeys(missing[:5]))
            raise KeyError(
                f"{len(missing)} prompts are missing from {self.path}; first entries: {preview}"
            )
        return output

    def __len__(self) -> int:
        return len(self.embeddings)


__all__ = ["TextEmbeddingLookup", "create_text_embedding_content_id"]
