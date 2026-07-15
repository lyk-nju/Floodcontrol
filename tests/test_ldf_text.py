from __future__ import annotations

import pytest
import torch

from utils.training.ldf.text import TextEmbeddingLookup


def _write_table(path, embeddings):
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": 8,
            "text_len": 4,
        },
        path,
    )


def test_text_embedding_lookup_reuses_frozen_cpu_tensors(tmp_path):
    path = tmp_path / "text.pt"
    _write_table(path, {"": torch.zeros(1, 8), "walk": torch.ones(2, 8)})
    lookup = TextEmbeddingLookup(path, expected_dim=8, expected_text_len=4)
    first, second = lookup.lookup(["walk", " walk "])
    assert first is second
    assert first.device.type == "cpu"


def test_text_embedding_lookup_fails_for_missing_or_nonfinite_caption(tmp_path):
    path = tmp_path / "text.pt"
    _write_table(path, {"": torch.zeros(1, 8), "bad": torch.full((1, 8), float("nan"))})
    with pytest.raises(ValueError, match="finite"):
        TextEmbeddingLookup(path, expected_dim=8, expected_text_len=4)

    _write_table(path, {"": torch.zeros(1, 8)})
    lookup = TextEmbeddingLookup(path, expected_dim=8, expected_text_len=4)
    with pytest.raises(KeyError, match="missing"):
        lookup.lookup(["walk"])
