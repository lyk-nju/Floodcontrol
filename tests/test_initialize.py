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
