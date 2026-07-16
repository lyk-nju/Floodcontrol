from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from tools.pretokenize_t5_text import collect_unique_captions
from utils.training.ldf.text import (
    TextEmbeddingLookup,
    create_text_embedding_content_id,
)


def test_pretokenize_collects_dataset_level_all_caption_inventory(tmp_path):
    texts = tmp_path / "texts"
    texts.mkdir()
    (tmp_path / "all.txt").write_text("train_sample\nprobe_sample\n")
    (texts / "train_sample.txt").write_text("training caption#token#0#0\n")
    (texts / "probe_sample.txt").write_text("probe caption#token#0#0\n")
    cfg = OmegaConf.create(
        {
            "data": {
                "text_meta_paths": [str(tmp_path / "all.txt")],
                "text_path": "texts",
            }
        }
    )

    assert collect_unique_captions(cfg) == {
        "",
        "training caption",
        "probe caption",
    }


def test_pretokenize_requires_dataset_level_all_inventory(tmp_path):
    cfg = OmegaConf.create(
        {
            "data": {
                "text_meta_paths": [str(tmp_path / "train.txt")],
                "text_path": "texts",
            }
        }
    )
    with pytest.raises(ValueError, match="all.txt"):
        collect_unique_captions(cfg)


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
