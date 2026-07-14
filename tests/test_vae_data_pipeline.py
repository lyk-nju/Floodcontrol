import json

import numpy as np
import torch

from datasets.strict4 import Strict4ArtifactDataset, collate_strict4
from tools.preprocess_strict4_smpl import process_file
from utils.conditions.vae import CONTRACT_VERSION


def test_native_rotation_npz_to_strict4_dataset(tmp_path):
    frames = 8
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(frames, 22, -1, -1).clone()
    translation = torch.zeros(frames, 3)
    translation[:, 0] = torch.arange(frames) / 20
    parents = torch.tensor([-1] + list(range(21)))
    offsets = torch.zeros(22, 3)
    offsets[1:, 1] = 0.05
    source = tmp_path / "native.npz"
    np.savez(
        source,
        local_rotations=rotations.numpy(),
        root_translation=translation.numpy(),
        parents=parents.numpy(),
        offsets=offsets.numpy(),
        fps=np.float32(20),
    )
    artifact = tmp_path / "artifacts" / "sample.npz"
    info = process_file(source, artifact, target_fps=20)
    assert info["frames"] == 8
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "name": "sample",
        "split": "train",
        "artifact": "artifacts/sample.npz",
        "contract_version": CONTRACT_VERSION,
    }) + "\n")
    dataset = Strict4ArtifactDataset(
        manifest_path=str(manifest), split="train", min_frames=8,
        max_frames=8, random_yaw=False
    )
    batch = collate_strict4([dataset[0], dataset[0]])
    assert batch["root_motion"].shape == (2, 8, 5)
    assert batch["body_motion"].shape == (2, 8, 265)
    assert batch["frame_valid_mask"].all()


def test_dataset_missing_manifest_fails_without_legacy_fallback(tmp_path):
    try:
        Strict4ArtifactDataset(manifest_path=str(tmp_path / "missing.jsonl"), split="train")
    except RuntimeError as error:
        assert "STRICT4_NATIVE_ROTATIONS_REQUIRED" in str(error)
    else:
        raise AssertionError("missing native manifest did not fail")
