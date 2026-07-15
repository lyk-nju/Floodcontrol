import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from models.vae_wan_1d import BodyVAE
from tools.motion_artifact import process_file
from tools.preprocess_humanml3d import build_dataset
from tools.pretokenize_body_latents import main as pretokenize_main
from utils.motion_representation import deterministic_sample_yaw
from utils.training.vae.checkpoint import (
    ema_state_dict,
    save_tokenizer_bundle,
    state_dict_sha256,
)


def _humanml263(frames: int = 8) -> np.ndarray:
    value = np.zeros((frames, 263), dtype=np.float32)
    value[:, 3] = 1.0
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    value[:, 67:193] = np.tile(identity, 21)
    return value


def _tiny_model() -> BodyVAE:
    return BodyVAE(
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        allow_identity_statistics=True,
        require_latent_statistics=False,
    )


def test_ema_state_dict_rejects_raw_only_and_replaces_every_parameter():
    model = _tiny_model()
    raw = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="missing ema_state"):
        ema_state_dict(model, {"state_dict": raw})
    shadows = [torch.full_like(parameter, 0.25) for parameter in model.parameters()]
    state = ema_state_dict(
        model,
        {"state_dict": raw, "ema_state": {"shadow_params": shadows}},
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(state[name], torch.full_like(parameter, 0.25))


def test_pretokenize_is_namespaced_two_pass_and_yaw_deterministic(tmp_path, monkeypatch):
    roots = []
    for dataset in ("HumanML3D_motion", "BABEL_motion"):
        root = tmp_path / dataset
        (root / "artifacts").mkdir(parents=True)
        source = tmp_path / f"{dataset}.npy"
        np.save(source, _humanml263())
        process_file(source, root / "artifacts" / "sample.npz", fps=20.0)
        (root / "train.txt").write_text("sample\n")
        roots.append(root)

    model = _tiny_model().eval()
    state = model.state_dict()
    inference_hash = state_dict_sha256(state)
    tokenizer = tmp_path / "tokenizer_ema.pt"
    save_tokenizer_bundle(
        model,
        tokenizer,
        model_config={
            "latent_dim": 4,
            "hidden_dim": 8,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "kernel_size": 3,
            "dropout": 0.0,
            "fps": 20.0,
        },
        checkpoint_metadata={
            "weights_kind": "ema",
            "training_checkpoint_sha256": "checkpoint",
            "inference_state_sha256": inference_hash,
            "motion_stats_sha256": "statistics",
            "global_step": 300000,
        },
    )
    output = tmp_path / "latents"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretokenize_body_latents.py",
            "--train-meta-paths",
            str(roots[0] / "train.txt"),
            str(roots[1] / "train.txt"),
            "--tokenizer",
            str(tokenizer),
            "--output",
            str(output),
            "--batch-size",
            "2",
            "--yaw-seed",
            "7",
        ],
    )
    pretokenize_main()
    for root in roots:
        artifact = output / "latents" / root.name / "sample.npz"
        assert artifact.is_file()
        with np.load(artifact, allow_pickle=False) as data:
            assert data["latent_motion"].shape == (2, 4)
            assert float(data["yaw_offset"]) == pytest.approx(
                deterministic_sample_yaw(root.name, "sample", seed=7), abs=1e-6
            )
    metadata = json.loads(str(np.load(output / "latent_stats.npz")["metadata"]))
    assert metadata["weights_kind"] == "ema"
    assert metadata["yaw_policy"] == "sample-id-sha256-uniform-v1"
    assert metadata["train_token_count"] == 4
    assert (output / "train.txt").read_text().splitlines() == [
        "HumanML3D_motion/sample",
        "BABEL_motion/sample",
    ]
