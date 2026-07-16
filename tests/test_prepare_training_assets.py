import json
from pathlib import Path

import numpy as np
import torch

from models.vae_wan_1d import BodyVAE
from tools.prepare_training_assets import main


def _humanml_motion(frames: int = 48) -> np.ndarray:
    motion = np.zeros((frames, 263), dtype=np.float32)
    motion[:, 1] = 0.05
    motion[:, 3] = 1.0
    positions = np.zeros((frames, 21, 3), dtype=np.float32)
    positions[..., 0] = np.linspace(0.1, 0.5, 21)
    positions[..., 1] = np.linspace(0.9, 1.8, 21)
    motion[:, 4:67] = positions.reshape(frames, -1)
    rotation = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 67:193] = np.tile(rotation, 21)
    motion[:, 259:263] = np.array([1, 0, 1, 0], dtype=np.float32)
    return motion


def _write_humanml_source(root: Path) -> None:
    (root / "new_joint_vecs").mkdir(parents=True)
    (root / "texts").mkdir()
    np.save(root / "new_joint_vecs" / "human.npy", _humanml_motion())
    (root / "texts" / "human.txt").write_text(
        "a person walks#person/NOUN walks/VERB#0#0\n"
    )
    for split in ("train", "val", "test"):
        (root / f"{split}.txt").write_text("human\n")


def _write_babel_source(root: Path) -> None:
    (root / "motions").mkdir(parents=True)
    (root / "texts").mkdir()
    np.save(root / "motions" / "babel.npy", _humanml_motion())
    (root / "texts" / "babel.txt").write_text(
        "walk forward#walk/VERB forward/ADV#0#2.4\n"
    )
    (root / "train_processed.txt").write_text("babel\n")
    (root / "val_processed.txt").write_text("babel\n")


def test_pre_vae_pipeline_builds_and_resumes_all_non_t5_assets(tmp_path):
    raw_data = tmp_path / "raw_data"
    _write_humanml_source(raw_data / "HumanML3D")
    _write_babel_source(raw_data / "BABEL_streamed")
    arguments = [
        "pre-vae",
        "--raw-data-root",
        str(raw_data),
        "--workers",
        "1",
        "--skip-t5",
    ]

    main(arguments)

    human = raw_data / "HumanML3D_motion"
    babel = raw_data / "BABEL_motion"
    assert (human / "artifacts" / "human.npz").is_file()
    assert (babel / "artifacts" / "babel.npz").is_file()
    for path, fields in (
        (
            human / "motion_stats.npz",
            {"local_root_mean", "local_root_std", "body_cont_mean", "body_cont_std"},
        ),
        (
            raw_data / "HumanML3D_BABEL_motion_stats.npz",
            {"local_root_mean", "local_root_std", "body_cont_mean", "body_cont_std"},
        ),
        (human / "root_stats.npz", {"root_mean", "root_std"}),
    ):
        with np.load(path, allow_pickle=False) as values:
            assert set(values.files) == fields
            for name in fields:
                assert np.isfinite(values[name]).all()
                if name.endswith("std"):
                    assert (values[name] > 0).all()

    report_path = raw_data / "training_assets.json"
    first_report = json.loads(report_path.read_text())
    assert first_report["stages"]["humanml_motion"]["status"] == "completed"
    assert first_report["stages"]["multi_motion_statistics"]["status"] == "completed"
    assert first_report["stages"]["pre_vae_verification"]["status"] == "completed"

    main(arguments)
    second_report = json.loads(report_path.read_text())
    for stage in (
        "humanml_motion",
        "babel_motion",
        "humanml_motion_statistics",
        "multi_motion_statistics",
        "ldf_root_statistics",
    ):
        assert second_report["stages"][stage]["status"] == "reused"


def test_post_vae_pipeline_uses_ema_checkpoint_and_writes_latent_stats(tmp_path):
    raw_data = tmp_path / "raw_data"
    _write_humanml_source(raw_data / "HumanML3D")
    _write_babel_source(raw_data / "BABEL_streamed")
    main(
        [
            "pre-vae",
            "--raw-data-root",
            str(raw_data),
            "--workers",
            "1",
            "--skip-t5",
        ]
    )

    motion_stats = raw_data / "HumanML3D_motion" / "motion_stats.npz"
    model = BodyVAE(
        motion_stats_path=motion_stats,
        latent_stats_path=None,
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
        dropout=0.0,
    )
    checkpoint = tmp_path / "vae" / "last.ckpt"
    checkpoint.parent.mkdir()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ema_state": {
                "shadow_params": [
                    parameter.detach().clone() for parameter in model.parameters()
                ]
            },
        },
        checkpoint,
    )
    config = tmp_path / "tiny_vae.yaml"
    config.write_text(
        "\n".join(
            [
                "model:",
                "  target: models.vae_wan_1d.BodyVAE",
                "  params:",
                "    latent_dim: 4",
                "    hidden_dim: 8",
                "    encoder_layers: 1",
                "    decoder_layers: 1",
                "    kernel_size: 3",
                "    dropout: 0.0",
                f"    motion_stats_path: {motion_stats}",
                "    latent_stats_path: null",
                "",
            ]
        )
    )

    main(
        [
            "post-vae",
            "--raw-data-root",
            str(raw_data),
            "--vae-config",
            str(config),
            "--vae-checkpoint",
            str(checkpoint),
            "--latent-device",
            "cpu",
            "--latent-batch-size",
            "1",
            "--skip-t5",
        ]
    )

    latent_stats = checkpoint.parent / "latent_stats.npz"
    with np.load(latent_stats, allow_pickle=False) as values:
        assert values["mean"].shape == (4,)
        assert values["std"].shape == (4,)
        assert np.isfinite(values["mean"]).all()
        assert (values["std"] > 0).all()
    report = json.loads((raw_data / "training_assets.json").read_text())
    assert report["stages"]["vae_latent_statistics"]["status"] == "completed"
    assert report["stages"]["final_verification"]["status"] == "completed"
