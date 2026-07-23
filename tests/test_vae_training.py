import sys
from pathlib import Path

import pytest

import train_vae
from train_vae import main as train_vae_main
from utils.initialize import load_config
from utils.training.vae import (
    VAELightningModule,
    VAELoss,
    create_dataset,
)
from tests.vae_helpers import make_vae
from utils.training.vae.lightning_module import validate_resume_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def test_train_vae_owns_runtime_assembly_but_not_model_implementation():
    source = (ROOT / "train_vae.py").read_text()
    assert "Trainer" in source
    assert "create_dataloaders" in source
    assert "VAELightningModule" in source
    assert "class VAELightningModule" not in source
    assert source.count("weights_only=False") == 2
    assert not (ROOT / "utils" / "training" / "vae" / "runner.py").exists()


def test_vae_training_package_exports_model_specific_components():
    assert VAELightningModule.__module__ == "utils.training.vae.lightning_module"
    assert VAELoss.__module__ == "utils.training.vae.losses"


def test_formal_vae_training_config_matches_frozen_recipe():
    cfg = load_config(str(ROOT / "configs" / "vae.yaml"))
    assert cfg.trainer.max_steps == 500_000
    assert cfg.trainer.devices == 1
    assert cfg.trainer.strategy == "auto"
    assert cfg.wandb_info.project == "VAE_Flood"
    assert cfg.wandb_info.entity == "laiyuankia-nanjing-university"
    assert cfg.wandb_info.key
    assert cfg.data.train_batch_size == 128
    assert cfg.data.val_batch_size == 128
    assert list(cfg.data.train_meta_paths) == [
        f"{cfg.dirs.raw_data}/HumanML3D_motion_local/train.txt"
    ]
    assert list(cfg.data.val_meta_paths) == [
        f"{cfg.dirs.raw_data}/HumanML3D_motion_local/val.txt"
    ]
    assert cfg.data.artifact_path == "artifacts"
    assert cfg.data.text_path is None
    assert "test_meta_paths" not in cfg.data
    assert "test_batch_size" not in cfg.data
    assert "test_steps" not in cfg.validation
    assert "collate_fn" not in cfg.data
    assert cfg.data.min_frames == 20
    assert cfg.data.max_frames == 200
    assert cfg.model.params.motion_stats_path == (
        f"{cfg.dirs.raw_data}/HumanML3D_motion_local/motion_stats.npz"
    )
    assert cfg.model.target == "models.vae_wan_1d.BodyVAE"
    assert "latent_stats_path" not in cfg.model.params
    assert "fps" not in cfg.model.params
    assert "allow_identity_statistics" not in cfg.model.params
    assert "require_latent_statistics" not in cfg.model.params
    assert cfg.loss.beta_kl == pytest.approx(1e-5)
    assert cfg.loss.kl_warmup_steps == 0
    assert cfg.loss.lambda_skating == pytest.approx(0.01)
    assert cfg.optimizer.target == "AdamW"
    assert cfg.optimizer.params.lr == pytest.approx(2e-4)
    assert cfg.lr_scheduler.target == "diffusers.optimization.get_constant_schedule_with_warmup"
    assert dict(cfg.lr_scheduler.params) == {"num_warmup_steps": 1_000}


def test_multi_vae_config_keeps_source_datasets_explicit():
    cfg = load_config(str(ROOT / "configs" / "vae_multi.yaml"))
    assert cfg.wandb_info.project == "VAE_Flood"
    assert cfg.data.target == "datasets.multi.MultiDataset"
    assert "collate_fn" not in cfg.data
    assert [entry.target for entry in cfg.data.datasets] == [
        "datasets.humanml3d.HumanML3DDataset",
        "datasets.babel.BABELDataset",
    ]
    assert [entry.text_path for entry in cfg.data.datasets] == [None, "texts"]
    assert cfg.model.params.motion_stats_path == (
        f"{cfg.dirs.raw_data}/HumanML3D_motion_local/motion_stats.npz"
    )
    assert "fps" not in cfg.model.params
    assert cfg.trainer.devices == 1
    assert cfg.trainer.strategy == "auto"
    assert cfg.data.train_batch_size == 128
    assert cfg.data.val_batch_size == 128


def test_basic_training_module_has_no_legacy_eval_or_hash_side_effects():
    source = (ROOT / "utils" / "training" / "lightning_module.py").read_text()
    assert "FLOODNET_DEBUG" not in source
    assert "ckpt_hash" not in source
    assert "TOKENIZERS_PARALLELISM" not in source
    assert "def test_step" not in source
    assert "initialize_metrics" not in source
    assert "ckpt_step_info" not in source
    assert "float(self.global_step + 1)" in source


def test_vae_dataset_builder_requires_preprocessed_motion_artifacts(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "vae.yaml"))
    cfg.config.data.train_meta_paths = [str(tmp_path / "missing.txt")]
    with pytest.raises(RuntimeError, match="split file not found"):
        create_dataset(cfg, "train")


def test_vae_entry_requires_motion_statistics_before_training(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_vae.py",
            "--config",
            str(ROOT / "configs" / "vae.yaml"),
            "--override",
            "model.params.motion_stats_path=null",
        ],
    )
    with pytest.raises(RuntimeError, match="MOTION_STATISTICS_REQUIRED"):
        train_vae_main()


def test_wandb_logger_uses_configured_wandb_info(tmp_path, monkeypatch):
    cfg = load_config(str(ROOT / "configs" / "vae.yaml"))
    captured = {}

    def fake_wandb_logger(**kwargs):
        captured.update(kwargs)
        return "wandb-logger"

    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(train_vae, "WandbLogger", fake_wandb_logger)
    logger = train_vae._create_logger(cfg, "test", tmp_path)
    assert logger == "wandb-logger"
    assert captured["project"] == cfg.wandb_info.project
    assert captured["entity"] == cfg.wandb_info.entity
    assert captured["name"] == "vae_body259_test"
    assert train_vae.os.environ["WANDB_API_KEY"] == cfg.wandb_info.key


def test_resume_checkpoint_rejects_statistics_before_loading():
    model = make_vae(
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    )
    checkpoint = {
        "state_dict": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
    }
    validate_resume_checkpoint(model, checkpoint)
    checkpoint["state_dict"]["body_cont_mean"][0] = 1.0
    with pytest.raises(RuntimeError, match="RESUME_STATISTICS_MISMATCH"):
        validate_resume_checkpoint(model, checkpoint)
