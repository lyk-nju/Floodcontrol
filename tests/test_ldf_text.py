from __future__ import annotations

import pytest
import torch

from utils.training.ldf.text import (
    TextEmbeddingLookup,
    create_text_embedding_content_id,
)


def _write_table(path, embeddings):
    content_id = create_text_embedding_content_id(
        embeddings,
        text_dim=8,
        text_len=4,
    )
    torch.save(
        {
            "embeddings": embeddings,
            "text_dim": 8,
            "text_len": 4,
            "content_id": content_id,
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
    torch.save(
        {
            "embeddings": {
                "": torch.zeros(1, 8),
                "bad": torch.full((1, 8), float("nan")),
            },
            "text_dim": 8,
            "text_len": 4,
            "content_id": "test-corrupt-table",
        },
        path,
    )
    lookup = TextEmbeddingLookup(path, expected_dim=8, expected_text_len=4)
    with pytest.raises(ValueError, match="finite"):
        lookup.lookup(["bad"])

    _write_table(path, {"": torch.zeros(1, 8)})
    lookup = TextEmbeddingLookup(path, expected_dim=8, expected_text_len=4)
    with pytest.raises(KeyError, match="missing"):
        lookup.lookup(["walk"])


def test_text_embedding_lookup_requires_offline_content_identity(tmp_path):
    path = tmp_path / "text.pt"
    torch.save(
        {"embeddings": {"": torch.zeros(1, 8)}, "text_dim": 8, "text_len": 4},
        path,
    )
    with pytest.raises(ValueError, match="content_id"):
        TextEmbeddingLookup(path, expected_dim=8, expected_text_len=4)


def test_text_embedding_content_identity_changes_with_tensor_contents():
    first = create_text_embedding_content_id(
        {"": torch.zeros(1, 8)}, text_dim=8, text_len=4
    )
    second = create_text_embedding_content_id(
        {"": torch.ones(1, 8)}, text_dim=8, text_len=4
    )
    assert first != second
