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

    human = raw_data / "HumanML3D_motion_local"
    babel = raw_data / "BABEL_motion_local"
    assert (human / "artifacts" / "human.npz").is_file()
    assert (babel / "artifacts" / "babel.npz").is_file()
    assert (human / "all.txt").read_text() == "human\n"
    assert (babel / "all.txt").read_text() == "babel\n"
    with np.load(human / "artifacts" / "human.npz") as human_motion, np.load(
        babel / "artifacts" / "babel.npz"
    ) as babel_motion:
        assert set(human_motion.files) == {
            "root_motion",
            "body_motion",
            "body_feature_valid_mask",
        }
        for name in human_motion.files:
            assert np.array_equal(human_motion[name], babel_motion[name])
        assert human_motion["root_motion"].dtype == np.float32
        assert human_motion["body_motion"].dtype == np.float32
        assert human_motion["body_feature_valid_mask"].dtype == np.bool_
        assert not human_motion["body_feature_valid_mask"][0, 189:259].any()
        assert human_motion["body_feature_valid_mask"][1:, 189:259].all()
        assert not human_motion["body_motion"][0, 189:259].any()
    fields = {"local_root_mean", "local_root_std", "body_cont_mean", "body_cont_std"}
    for path in (human / "motion_stats.npz",):
        with np.load(path, allow_pickle=False) as values:
            assert fields <= set(values.files)
            assert set(values.files) - fields <= {"metadata"}
            for name in fields:
                assert np.isfinite(values[name]).all()
                if name.endswith("std"):
                    assert (values[name] > 0).all()

    assert not (raw_data / "HumanML3D_BABEL_motion_stats.npz").exists()
    report_path = raw_data / "training_assets_local.json"
    first_report = json.loads(report_path.read_text())
    assert first_report["stages"]["humanml_motion"]["status"] == "completed"
    assert first_report["stages"]["pre_vae_verification"]["status"] == "completed"

    main(arguments)
    second_report = json.loads(report_path.read_text())
    for stage in (
        "humanml_motion",
        "babel_motion",
        "humanml_motion_statistics",
    ):
        assert second_report["stages"][stage]["status"] == "reused"

    # Old Body259 artifacts copied HumanML's forward contact without marking
    # the true sequence start invalid.  The resumable pipeline must detect and
    # rebuild that semantic mismatch even though shape and dtype still match.
    artifact = human / "artifacts" / "human.npz"
    with np.load(artifact, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]).copy() for name in values.files}
    arrays["body_feature_valid_mask"][0, 255:259] = True
    np.savez_compressed(artifact, **arrays)
    main(arguments)
    third_report = json.loads(report_path.read_text())
    assert third_report["stages"]["humanml_motion"]["status"] == "completed"
    assert third_report["stages"]["babel_motion"]["status"] == "reused"
    assert (
        third_report["stages"]["humanml_motion_statistics"]["status"]
        == "reused"
    )


def test_verify_uses_a_self_contained_ema_vae_checkpoint(tmp_path):
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

    motion_stats = raw_data / "HumanML3D_motion_local" / "motion_stats.npz"
    model = BodyVAE(
        motion_stats_path=motion_stats,
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
    config = tmp_path / "tiny_ldf.yaml"
    config.write_text(
        "\n".join(
            [
                f"base_config: {Path(__file__).resolve().parents[1] / 'configs' / 'ldf_base.yaml'}",
                "vae:",
                "  params:",
                "    latent_dim: 4",
                "    hidden_dim: 8",
                "    encoder_layers: 1",
                "    decoder_layers: 1",
                "    kernel_size: 3",
                "    dropout: 0.0",
                "",
            ]
        )
    )

    main(
        [
            "verify",
            "--raw-data-root",
            str(raw_data),
            "--vae-checkpoint",
            str(checkpoint),
            "--ldf-config",
            str(config),
            "--ldf-multi-config",
            str(config),
            "--skip-t5",
        ]
    )

    report = json.loads((raw_data / "training_assets_local.json").read_text())
    assert report["stages"]["final_verification"]["status"] == "completed"
    details = report["stages"]["final_verification"]["details"]
    assert details["vae_checkpoint"]["latent_dim"] == 4
    assert set(details["vae_checkpoint"]["physical_statistics"]) == {
        "body_cont_mean",
        "body_cont_std",
        "local_root_mean",
        "local_root_std",
    }
