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
    assert cfg.trainer.max_steps == 300_000
    assert cfg.trainer.devices == 3
    assert cfg.trainer.strategy == "ddp"
    assert cfg.wandb_info.project == "VAE_Flood"
    assert cfg.wandb_info.entity == "laiyuankia-nanjing-university"
    assert cfg.wandb_info.key
    assert cfg.data.train_bs == 32
    assert list(cfg.data.train_meta_paths) == [
        f"{cfg.dirs.raw_data}/HumanML3D_motion/train.txt"
    ]
    assert list(cfg.data.val_meta_paths) == [
        f"{cfg.dirs.raw_data}/HumanML3D_motion/val.txt"
    ]
    assert cfg.data.artifact_path == "artifacts"
    assert cfg.data.min_frames == 20
    assert cfg.data.max_frames == 200
    assert cfg.model.params.motion_stats_path == (
        f"{cfg.dirs.raw_data}/HumanML3D_motion/motion_stats.npz"
    )
    assert cfg.loss.beta_kl == pytest.approx(1e-5)
    assert cfg.loss.kl_warmup_steps == 0
    assert cfg.loss.lambda_skating == pytest.approx(0.01)
    assert cfg.optimizer.target == "AdamW"
    assert cfg.optimizer.params.lr == pytest.approx(2e-4)
    assert cfg.lr_scheduler.target == "diffusers.optimization.get_constant_schedule_with_warmup"
    assert dict(cfg.lr_scheduler.params) == {"num_warmup_steps": 1_000}


def test_multi_vae_config_keeps_source_datasets_explicit():
    cfg = load_config(str(ROOT / "configs" / "vae_multi.yaml"))
    assert cfg.data.target == "datasets.multi.MultiDataset"
    assert cfg.data.collate_fn == "datasets.multi.collate_multi"
    assert [entry.target for entry in cfg.data.datasets] == [
        "datasets.humanml3d.HumanML3DDataset",
        "datasets.babel.BABELDataset",
    ]
    assert cfg.model.params.motion_stats_path == (
        f"{cfg.dirs.raw_data}/HumanML3D_BABEL_motion_stats.npz"
    )


def test_vae_dataset_builder_requires_preprocessed_motion_artifacts(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "vae.yaml"))
    cfg.config.data.train_meta_paths = [str(tmp_path / "missing.txt")]
    with pytest.raises(RuntimeError, match="MOTION_ARTIFACT_DATA_REQUIRED"):
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
    assert captured["name"] == "vae_body265_test"
    assert train_vae.os.environ["WANDB_API_KEY"] == cfg.wandb_info.key
