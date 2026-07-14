from pathlib import Path

import pytest

from models.diffusion_forcing_wan import LDF
from models.vae_wan_1d import BodyVAE
from train_ldf import TRAINING_MIGRATION_ERROR, main as train_main
from utils.initialize import instantiate, load_config
from web_demo.model_manager import WEB_MIGRATION_ERROR, get_model_manager


ROOT = Path(__file__).resolve().parents[1]


def test_tiny_core_config_instantiates_public_ldf():
    cfg = load_config(str(ROOT / "configs" / "ldf.yaml"))
    model = instantiate(cfg.model.target, cfg=None, **cfg.model.params)
    assert isinstance(model, LDF)


def test_tiny_vae_config_instantiates_public_body_vae():
    cfg = load_config(str(ROOT / "configs" / "vae.yaml"))
    params = dict(cfg.model.params)
    params.update(
        motion_stats_path=None,
        latent_stats_path=None,
        allow_identity_statistics=True,
        require_latent_statistics=False,
    )
    model = instantiate(cfg.model.target, cfg=None, **params)
    assert isinstance(model, BodyVAE)


def test_legacy_vae_config_and_class_are_removed():
    assert not (ROOT / "configs" / "vae_wan_1d.yaml").exists()
    import models.vae_wan_1d as module

    assert not hasattr(module, "VAEWanModel")


def test_training_entry_is_explicitly_blocked():
    with pytest.raises(RuntimeError, match="BLOCKED_ON_BODY_VAE"):
        train_main()
    assert "four-frame body VAE" in TRAINING_MIGRATION_ERROR


def test_web_entry_is_explicitly_blocked():
    with pytest.raises(RuntimeError, match="BLOCKED_ON_BODY_VAE"):
        get_model_manager()
    assert "four-frame body VAE" in WEB_MIGRATION_ERROR


@pytest.mark.parametrize(
    "relative_path",
    [
        "models/diffusion_forcing_wan_tiny.py",
        "models/tools/wan_controlnet.py",
        "models/tools/traj_encoder.py",
        "models/root_" + "refiner.py",
        "utils/conditions/root_" + "refiner.py",
        "utils/inference/root_plan.py",
        "utils/inference/stream_generator.py",
    ],
)
def test_removed_architecture_files_are_physically_absent(relative_path):
    assert not (ROOT / relative_path).exists()
