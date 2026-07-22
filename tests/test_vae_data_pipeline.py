import sys

import numpy as np
import pytest
import torch

from datasets.humanml3d import HumanML3DDataset
from tools.compute_vae_stats import compute_motion_statistics, main as compute_stats_main
from tools.convert_motion_263_to_259 import convert_motion_263_to_259, recover_root_263
from tools.preprocess_humanml3d import build_dataset
from utils.motion_process import rotate_motion_yaw, unpack_body
from utils.training.vae.data import VAEWindowCollator


def make_humanml263(frames: int = 8) -> np.ndarray:
    feature = np.zeros((frames, 263), dtype=np.float32)
    feature[:, 1] = 0.05
    feature[:, 3] = 1.0
    positions = np.zeros((frames, 21, 3), dtype=np.float32)
    positions[..., 0] = np.linspace(0.1, 0.5, 21)
    positions[..., 1] = np.linspace(0.9, 1.8, 21)
    feature[:, 4:67] = positions.reshape(frames, -1)
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    feature[:, 67:193] = np.tile(identity_6d, 21)
    feature[:, 259:263] = np.array([1, 0, 1, 0], dtype=np.float32)
    return feature


def write_processed_sample(root, name="sample", frames=12, *, legacy_metadata=True):
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    root_motion = np.zeros((frames, 5), dtype=np.float32)
    root_motion[:, 0] = np.arange(frames) + 10
    root_motion[:, 1] = 1
    root_motion[:, 2] = np.arange(frames) * 2 + 20
    root_motion[:, 3] = 1
    body = np.zeros((frames, 259), dtype=np.float32)
    body[:, 0] = np.arange(frames)
    valid = np.ones_like(body, dtype=bool)
    payload = {
        "root_motion": root_motion,
        "body_motion": body,
        "body_feature_valid_mask": valid,
    }
    if legacy_metadata:
        payload.update(
            contract_version="body259-v1",
            converter_version="old-converter",
            source_sha256="ignored",
            fps=np.float32(123.0),
        )
    np.savez_compressed(root / "artifacts" / f"{name}.npz", **payload)
    return root_motion, body, valid


def test_humanml263_conversion_recovers_root_body_and_backward_velocity():
    root, body, valid = convert_motion_263_to_259(
        torch.from_numpy(make_humanml263()), fps=20
    )
    assert root.shape == (8, 5)
    assert body.shape == (8, 259)
    assert torch.allclose(root[:, 0], torch.arange(8) * 0.05, atol=1e-6)
    parts = unpack_body(body)
    assert torch.allclose(parts["joint_velocities"][1:, 0, 0], torch.ones(7))
    assert not valid[0, 189:255].any()
    assert valid[1:, 189:255].all()


def test_humanml263_root_recovery_preserves_floating_dtype():
    heading, root = recover_root_263(torch.from_numpy(make_humanml263()).double())
    assert heading.dtype == torch.float64
    assert root.dtype == torch.float64


def test_humanml_dataset_returns_full_clip_parses_text_and_ignores_old_metadata(tmp_path):
    root_motion, body, valid = write_processed_sample(tmp_path, frames=12)
    (tmp_path / "train.txt").write_text("sample\n")
    texts = tmp_path / "source_texts"
    texts.mkdir()
    (texts / "sample.txt").write_text(
        "whole motion#whole/NOUN motion/NOUN#0#0\n"
        "middle#middle/NOUN#0.2#0.4\n"
    )
    dataset = HumanML3DDataset(
        meta_paths=[tmp_path / "train.txt"],
        split="train",
        text_path=texts,
        fps=20,
    )
    assert set(dataset.dataset[0]) == {
        "dataset", "name", "motion_path", "text_path"
    }
    sample = dataset[0]
    assert sample["dataset"] == "HumanML3D"
    assert set(sample) == {
        "dataset", "name", "root_motion", "body_motion",
        "body_feature_valid_mask", "text_data",
    }
    assert torch.equal(sample["root_motion"], torch.from_numpy(root_motion))
    assert torch.equal(sample["body_motion"], torch.from_numpy(body))
    assert torch.equal(sample["body_feature_valid_mask"], torch.from_numpy(valid))
    assert sample["text_data"] == [
        {
            "text": "whole motion",
            "tokens": ["whole/NOUN", "motion/NOUN"],
            "start_frame": 0,
            "end_frame": 12,
        },
        {
            "text": "middle",
            "tokens": ["middle/NOUN"],
            "start_frame": 4,
            "end_frame": 8,
        },
    ]


def test_humanml_dataset_ignores_annotations_without_positive_motion_overlap(
    tmp_path,
):
    write_processed_sample(tmp_path, frames=12)
    (tmp_path / "train.txt").write_text("sample\n")
    texts = tmp_path / "texts"
    texts.mkdir()
    (texts / "sample.txt").write_text(
        "whole#whole/NOUN#0#0\n"
        "zero duration#zero/NOUN#0.2#0.2\n"
        "reversed#reverse/VERB#1.0#0.4\n"
        "past motion#past/NOUN#2.0#3.0\n"
    )
    dataset = HumanML3DDataset(
        meta_paths=[tmp_path / "train.txt"],
        split="train",
        text_path="texts",
        fps=20,
    )

    assert dataset[0]["text_data"] == [
        {
            "text": "whole",
            "tokens": ["whole/NOUN"],
            "start_frame": 0,
            "end_frame": 12,
        }
    ]


def test_vae_validation_collator_uses_prefix_rebase_previous_root_and_padding(tmp_path):
    write_processed_sample(tmp_path, "long", frames=12)
    write_processed_sample(tmp_path, "short", frames=8)
    (tmp_path / "val.txt").write_text("long\nshort\n")
    dataset = HumanML3DDataset(meta_paths=[tmp_path / "val.txt"], split="val")
    batch = VAEWindowCollator(
        min_frames=4, max_frames=12, training=False, random_yaw=True
    )([dataset[0], dataset[1]])
    assert batch["root_motion"].shape == (2, 12, 5)
    assert batch["body_motion"].shape == (2, 12, 259)
    assert batch["frame_valid_mask"][0].all()
    assert batch["frame_valid_mask"][1, :8].all()
    assert not batch["frame_valid_mask"][1, 8:].any()
    assert torch.equal(batch["root_motion"][:, 0, [0, 2]], torch.zeros(2, 2))
    assert not batch["previous_root_valid_mask"].any()
    assert not batch["body_feature_valid_mask"][1, 8:].any()


def test_vae_training_collator_owns_random_aligned_crop_yaw_and_boundary(monkeypatch):
    root = torch.zeros(16, 5)
    root[:, 0] = torch.arange(16, dtype=torch.float32)
    root[:, 2] = torch.arange(16, dtype=torch.float32) * 2
    root[:, 3] = 1
    body = torch.zeros(16, 259)
    body[:, 0] = 1
    sample = {
        "dataset": "d",
        "name": "n",
        "root_motion": root,
        "body_motion": body,
        "body_feature_valid_mask": torch.ones_like(body, dtype=torch.bool),
        "text_data": [],
    }
    choices = iter((2, 1))  # two tokens, starting at token one
    monkeypatch.setattr("utils.training.vae.data.random.randint", lambda *_: next(choices))
    monkeypatch.setattr(torch, "rand", lambda *_, **__: torch.tensor([0.25]))
    batch = VAEWindowCollator(
        min_frames=8, max_frames=12, training=True, random_yaw=True
    )([sample])
    rebased_root = root[4:12].clone()
    rebased_root[:, 0] -= root[4, 0]
    rebased_root[:, 2] -= root[4, 2]
    previous = root[3].clone()
    previous[0] -= root[4, 0]
    previous[2] -= root[4, 2]
    expected_root, expected_body = rotate_motion_yaw(
        rebased_root[None], body[4:12][None], torch.tensor([torch.pi / 2])
    )
    assert torch.allclose(batch["root_motion"], expected_root, atol=1e-6)
    assert torch.allclose(batch["body_motion"], expected_body, atol=1e-6)
    assert batch["previous_root_valid_mask"].item()
    assert batch["previous_root_frame"].shape == (1, 5)
    assert torch.allclose(batch["previous_root_frame"][0, :3].norm(), previous[:3].norm())


def test_preprocess_writes_minimal_npz_split_and_copies_text(tmp_path):
    source = tmp_path / "HumanML3D"
    (source / "new_joint_vecs").mkdir(parents=True)
    (source / "texts").mkdir()
    np.save(source / "new_joint_vecs" / "sample.npy", make_humanml263(20))
    (source / "texts" / "sample.txt").write_text("motion#motion/NOUN#0#0\n")
    for split in ("train", "val", "test"):
        (source / f"{split}.txt").write_text("sample\n")
    output = tmp_path / "HumanML3D_motion"
    summary = build_dataset(source, output, workers=1)
    assert summary["copied_texts"] == 1
    assert (output / "all.txt").read_text() == "sample\n"
    assert (output / "texts" / "sample.txt").read_text().startswith("motion#")
    with np.load(output / "artifacts" / "sample.npz", allow_pickle=False) as data:
        assert set(data.files) == {
            "root_motion", "body_motion", "body_feature_valid_mask"
        }
    resumed = build_dataset(source, output, workers=1)
    assert resumed["converted"] == 0
    assert resumed["skipped"] == 1


def test_statistics_consumes_dataset_samples_and_cli_smoke(tmp_path, monkeypatch):
    write_processed_sample(tmp_path, frames=12)
    (tmp_path / "train.txt").write_text("sample\n")
    dataset = HumanML3DDataset(meta_paths=[tmp_path / "train.txt"], split="train")
    statistics = compute_motion_statistics(dataset)
    assert statistics["body_cont_mean"].shape == (255,)
    assert statistics["local_root_mean"].shape == (4,)

    output = tmp_path / "stats.npz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_vae_stats.py", "--train-meta-paths", str(tmp_path / "train.txt"),
            "--output", str(output),
        ],
    )
    compute_stats_main()
    assert output.is_file()


def test_dataset_rejects_duplicate_sample_ids_within_namespace(tmp_path):
    write_processed_sample(tmp_path)
    first = tmp_path / "train.txt"
    second = tmp_path / "extra.txt"
    first.write_text("sample\n")
    second.write_text("sample\n")
    with pytest.raises(ValueError, match="duplicate sample id"):
        HumanML3DDataset(meta_paths=[first, second], split="train")


def test_humanml_source_identity_does_not_depend_on_directory_name(tmp_path):
    renamed_root = tmp_path / "arbitrary_release_name"
    write_processed_sample(renamed_root)
    (renamed_root / "train.txt").write_text("sample\n")
    dataset = HumanML3DDataset(
        meta_paths=[renamed_root / "train.txt"], split="train"
    )
    assert dataset[0]["dataset"] == "HumanML3D"
