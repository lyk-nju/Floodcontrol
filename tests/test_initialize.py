from pathlib import Path

import pytest

from utils.initialize import (
    ProjectConfig,
    instantiate_target,
    load_config,
    resolve_function,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_config_merges_paths_and_parses_scalar_overrides():
    config = load_config(
        ROOT / "configs" / "vae.yaml",
        {
            "seed": "17",
            "train": "false",
            "resume_ckpt": "null",
            "model.params.dropout": "0.25",
        },
    )
    assert isinstance(config, ProjectConfig)
    assert config.seed == 17
    assert config.train is False
    assert config.resume_ckpt is None
    assert config.model.params.dropout == pytest.approx(0.25)


def test_dynamic_targets_are_explicitly_validated():
    value = instantiate_target("builtins.dict", cfg=None, answer=42)
    assert value == {"answer": 42}
    function = resolve_function("utils.token_frame.token_count_to_frame_count")
    assert function(3) == 12

    with pytest.raises(ValueError, match="fully-qualified"):
        instantiate_target("dict")
    with pytest.raises(AttributeError, match="does not exist"):
        resolve_function("utils.token_frame.missing_function")
    with pytest.raises(TypeError, match="not callable"):
        resolve_function("utils.token_frame.FRAMES_PER_TOKEN")


def test_config_base_is_relative_recursive_and_removed_from_result(tmp_path):
    shared = tmp_path / "shared.yaml"
    base = tmp_path / "base.yaml"
    child = tmp_path / "experiments" / "child.yaml"
    child.parent.mkdir()
    shared.write_text("model:\n  hidden: 128\n  layers: 4\n", encoding="utf-8")
    base.write_text(
        "base_config: shared.yaml\nmodel:\n  layers: 6\nseed: 11\n",
        encoding="utf-8",
    )
    child.write_text(
        "base_config: ../base.yaml\nmodel:\n  hidden: 256\n",
        encoding="utf-8",
    )

    config = load_config(child, {"seed": "17"})
    assert config.model.hidden == 256
    assert config.model.layers == 6
    assert config.seed == 17
    assert "base_config" not in config.config


def test_config_base_cycle_is_rejected(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("base_config: second.yaml\n", encoding="utf-8")
    second.write_text("base_config: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="base cycle"):
        load_config(first)
