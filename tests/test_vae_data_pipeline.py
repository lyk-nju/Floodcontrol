import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from datasets.humanml3d import HumanML3DDataset, collate_humanml3d
from tools.compute_vae_stats import main as compute_stats_main
from tools.preprocess_humanml3d import build_dataset, process_file
from utils.training.vae.data import validate_training_statistics
from utils.motion_representation import (
    HUMANML_SOURCE_REPRESENTATION,
    MOTION_CONVERTER_VERSION,
    derive_patched_local_root,
    humanml263_to_root_body_motion,
    rotate_root_body_yaw,
    rotate_root_yaw,
    unpack_body_motion,
)


def make_humanml263(frames: int = 8) -> np.ndarray:
    feature = np.zeros((frames, 263), dtype=np.float32)
    feature[:, 1] = 0.05  # 1 m/s local root x velocity at 20 FPS
    feature[:, 3] = 1.0
    positions = np.zeros((frames, 21, 3), dtype=np.float32)
    positions[..., 0] = np.linspace(0.1, 0.5, 21)
    positions[..., 1] = np.linspace(0.9, 1.8, 21)
    feature[:, 4:67] = positions.reshape(frames, -1)
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    feature[:, 67:193] = np.tile(identity_6d, 21)
    feature[:, 259:263] = np.array([1, 0, 1, 0], dtype=np.float32)
    return feature


def test_humanml263_conversion_recovers_root_body_and_backward_velocity():
    root, body, valid = humanml263_to_root_body_motion(
        torch.from_numpy(make_humanml263()), fps=20
    )
    assert root.shape == (8, 5)
    assert body.shape == (8, 265)
    assert torch.allclose(root[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert torch.allclose(root[:, 0], torch.arange(8) * 0.05, atol=1e-6)
    parts = unpack_body_motion(body)
    assert torch.allclose(parts["joint_velocities"][1:, 0, 0], torch.ones(7))
    assert not valid[0, 195:261].any()
    assert valid[1:, 195:261].all()
    assert torch.equal(
        parts["foot_contacts"],
        torch.tensor([[1.0, 0.0, 1.0, 0.0]]).expand(8, -1),
    )


def test_humanml_root_heading_and_ik_root_rotation_keep_their_conventions():
    feature = make_humanml263()
    feature[0, 0] = 0.1
    root, body, _ = humanml263_to_root_body_motion(torch.from_numpy(feature))
    parts = unpack_body_motion(body)
    root_rotation = parts["joint_rotations_6d"][1, 0]
    assert torch.allclose(
        root[1, 3:], torch.tensor([torch.cos(torch.tensor(-0.2)), torch.sin(torch.tensor(-0.2))]),
        atol=1e-6,
    )
    assert torch.allclose(
        root_rotation,
        torch.tensor([torch.cos(torch.tensor(0.2)), 0.0, -torch.sin(torch.tensor(0.2)), 0.0, 1.0, 0.0]),
        atol=1e-6,
    )


def test_humanml263_to_motion_dataset(tmp_path):
    source = tmp_path / "sample.npy"
    np.save(source, make_humanml263())
    artifact = tmp_path / "artifacts" / "sample.npz"
    info = process_file(source, artifact, fps=20)
    assert info["frames"] == 8
    assert info["source_representation"] == "humanml3d-263-ik-v1"
    train_meta = tmp_path / "train.txt"
    train_meta.write_text("sample\n")
    dataset = HumanML3DDataset(
        meta_paths=[str(train_meta)], split="train", min_frames=8,
        max_frames=8, random_yaw=False
    )
    batch = collate_humanml3d([dataset[0], dataset[0]])
    assert batch["root_motion"].shape == (2, 8, 5)
    assert batch["body_motion"].shape == (2, 8, 265)
    assert batch["frame_valid_mask"].all()


def test_training_random_yaw_rotates_every_world_space_feature(tmp_path, monkeypatch):
    source = tmp_path / "sample.npy"
    np.save(source, make_humanml263())
    artifact = tmp_path / "artifacts" / "sample.npz"
    process_file(source, artifact, fps=20)
    train_meta = tmp_path / "train.txt"
    train_meta.write_text("sample\n")
    canonical = HumanML3DDataset(
        meta_paths=[str(train_meta)], split="val", min_frames=8,
        max_frames=8, random_yaw=False
    )[0]
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.25]))
    augmented = HumanML3DDataset(
        meta_paths=[str(train_meta)], split="train", min_frames=8,
        max_frames=8, random_yaw=True
    )[0]
    expected_root, expected_body = rotate_root_body_yaw(
        canonical["root_motion"][None],
        canonical["body_motion"][None],
        torch.tensor([torch.pi / 2]),
    )
    assert torch.allclose(augmented["root_motion"], expected_root[0], atol=1e-6)
    assert torch.allclose(augmented["body_motion"], expected_body[0], atol=1e-6)
    assert torch.allclose(
        augmented["root_motion"][0, 3:], torch.tensor([0.0, 1.0]), atol=1e-6
    )
    before, before_valid = derive_patched_local_root(
        canonical["root_motion"][None], None
    )
    after, after_valid = derive_patched_local_root(
        augmented["root_motion"][None], None
    )
    assert torch.equal(before_valid, after_valid)
    assert torch.allclose(before[before_valid], after[after_valid], atol=1e-5)


def test_validation_disables_random_yaw_even_when_config_enables_it(tmp_path, monkeypatch):
    source = tmp_path / "sample.npy"
    np.save(source, make_humanml263())
    process_file(source, tmp_path / "artifacts" / "sample.npz", fps=20)
    meta = tmp_path / "val.txt"
    meta.write_text("sample\n")
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([0.25]))
    canonical = HumanML3DDataset(
        meta_paths=[str(meta)], split="val", min_frames=8,
        max_frames=8, random_yaw=False,
    )[0]
    configured = HumanML3DDataset(
        meta_paths=[str(meta)], split="val", min_frames=8,
        max_frames=8, random_yaw=True,
    )[0]
    assert torch.equal(configured["root_motion"], canonical["root_motion"])
    assert torch.equal(configured["body_motion"], canonical["body_motion"])


def test_dataset_missing_split_metadata_fails_without_online_conversion(tmp_path):
    try:
        HumanML3DDataset(meta_paths=[str(tmp_path / "missing.txt")], split="train")
    except RuntimeError as error:
        assert "MOTION_ARTIFACT_DATA_REQUIRED" in str(error)
    else:
        raise AssertionError("missing motion split metadata did not fail")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split": "trian"}, "unsupported split"),
        ({"split": "train", "min_frames": 21}, "positive multiple of four"),
        ({"split": "train", "max_frames": 202}, "positive multiple of four"),
        ({"split": "train", "expected_fps": 0}, "finite and positive"),
    ],
)
def test_dataset_rejects_invalid_contract_configuration(tmp_path, kwargs, message):
    options = {"split": "train", "min_frames": 20, "max_frames": 200}
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        HumanML3DDataset(
            meta_paths=[str(tmp_path / "not-read.txt")],
            **options,
        )


def test_dataset_rejects_malformed_or_incompatible_artifacts(tmp_path):
    source = tmp_path / "sample.npy"
    np.save(source, make_humanml263())
    artifact = tmp_path / "artifacts" / "sample.npz"
    process_file(source, artifact, fps=20)
    train_meta = tmp_path / "train.txt"
    train_meta.write_text("sample\n")
    with np.load(artifact, allow_pickle=False) as data:
        valid_payload = {name: np.asarray(data[name]) for name in data.files}

    def assert_rejected(overrides, message):
        np.savez_compressed(artifact, **{**valid_payload, **overrides})
        dataset = HumanML3DDataset(
            meta_paths=[str(train_meta)],
            split="val",
            min_frames=4,
            max_frames=8,
        )
        with pytest.raises(ValueError, match=message):
            dataset[0]

    assert_rejected(
        {"root_motion": valid_payload["root_motion"][:-1]},
        "lengths differ",
    )
    assert_rejected(
        {
            "root_motion": valid_payload["root_motion"][:-1],
            "body_motion": valid_payload["body_motion"][:-1],
            "body_feature_valid_mask": valid_payload["body_feature_valid_mask"][:-1],
        },
        "divisible by four",
    )
    assert_rejected(
        {"body_feature_valid_mask": valid_payload["body_feature_valid_mask"][:, :-1]},
        "must match body_motion",
    )
    assert_rejected({"fps": np.float32(30)}, "FPS mismatch")
    assert_rejected({"source_representation": "other-source"}, "source representation")
    non_finite = valid_payload["body_motion"].copy()
    non_finite[0, 0] = np.nan
    assert_rejected({"body_motion": non_finite}, "non-finite")
    assert_rejected({"previous_root_frame": np.zeros(4, dtype=np.float32)}, "shape")


def test_root_only_yaw_rotation_matches_joint_root_rotation():
    root, body, _ = humanml263_to_root_body_motion(
        torch.from_numpy(make_humanml263())
    )
    angle = torch.tensor([0.73])
    root_only = rotate_root_yaw(root[None], angle)
    root_with_body, _ = rotate_root_body_yaw(root[None], body[None], angle)
    assert torch.allclose(root_only, root_with_body)
    assert HUMANML_SOURCE_REPRESENTATION == "humanml3d-263-ik-v1"


def test_preprocess_and_statistics_use_humanml_split_contract(tmp_path, monkeypatch):
    source_root = tmp_path / "HumanML3D"
    motion_root = source_root / "new_joint_vecs"
    motion_root.mkdir(parents=True)
    np.save(motion_root / "sample.npy", make_humanml263(frames=20))
    np.save(motion_root / "short.npy", make_humanml263(frames=8))
    invalid = make_humanml263(frames=20)
    invalid[0, 0] = np.nan
    np.save(motion_root / "invalid.npy", invalid)
    for split in ("train", "val", "test"):
        values = "sample\nshort\ninvalid\n" if split == "train" else "sample\n"
        (source_root / f"{split}.txt").write_text(values)
    output = tmp_path / "HumanML3D_motion"
    summary = build_dataset(source_root, output, workers=1)
    assert summary["splits"] == {"train": 1, "val": 1, "test": 1}
    assert summary["too_short"] == {"train": 1, "val": 0, "test": 0}
    assert summary["invalid_nonfinite"] == {"train": 1, "val": 0, "test": 0}
    assert summary["converted"] == 1
    assert summary["skipped"] == 0
    resumed = build_dataset(source_root, output, workers=1)
    assert resumed["converted"] == 0
    assert resumed["skipped"] == 1
    assert not list((output / "artifacts").glob("*.tmp.npz"))
    for split in ("train", "val", "test"):
        assert (output / f"{split}.txt").read_text() == "sample\n"
    with np.load(output / "artifacts" / "sample.npz", allow_pickle=False) as data:
        assert str(data["contract_version"]) == "body265-v1"
        assert str(data["converter_version"]) == "humanml265"
        assert str(data["source_representation"]) == "humanml3d-263-ik-v1"

    # A source change invalidates the cached artifact instead of silently
    # retaining stale converted tensors.
    changed = make_humanml263(frames=20)
    changed[:, 3] = 1.1
    np.save(motion_root / "sample.npy", changed)
    rebuilt = build_dataset(source_root, output, workers=1)
    assert rebuilt["converted"] == 1
    assert rebuilt["skipped"] == 0

    stats_path = output / "motion_stats.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_vae_stats.py",
            "--train-meta-paths", str(output / "train.txt"),
            "--output", str(stats_path),
        ],
    )
    compute_stats_main()
    with np.load(stats_path, allow_pickle=False) as stats:
        assert stats["body_cont_mean"].shape == (261,)
        assert stats["local_root_mean"].shape == (4,)
        assert "global_root_mean" not in stats
        assert "global_root_std" not in stats
        metadata = json.loads(str(stats["metadata"]))
        assert metadata["converter_version"] == MOTION_CONVERTER_VERSION
        assert len(metadata["artifact_manifest_sha256"]) == 64
        assert metadata["source_representations"] == ["humanml3d-263-ik-v1"]
        assert metadata["yaw_statistics"] == "uniform-four-point-quadrature-v1"

    dataset = HumanML3DDataset(
        meta_paths=[str(output / "train.txt")],
        split="train",
        min_frames=20,
        max_frames=20,
    )
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            params=SimpleNamespace(
                motion_stats_path=str(stats_path),
                fps=20.0,
            )
        )
    )
    validate_training_statistics(cfg, dataset)

    # Rebuilding one artifact after its source changes invalidates statistics
    # even though train.txt itself is unchanged.
    changed[:, 3] = 1.2
    np.save(motion_root / "sample.npy", changed)
    build_dataset(source_root, output, workers=1)
    with pytest.raises(RuntimeError, match="VAE_STATISTICS_STALE"):
        validate_training_statistics(cfg, dataset)
