from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from datasets.humanml3d import HumanML3DDataset
from models.vae_wan_1d import BodyVAE
from tests.vae_helpers import write_statistics
from utils.conditions.ldf import LDFCondition
from utils.training.ldf.batch import anchor_physical_batch
from utils.training.ldf.losses import compute_velocity_loss
from utils.training.ldf.data import LDFSpanCollator
from utils.training.ldf.lightning_module import LDFLightningModule
from utils.training.ldf.self_forcing import (
    SelfForcingState,
    run_self_forcing_rollout,
    sample_window_plan,
)
from utils.training.ldf.text import create_text_embedding_content_id


def _write_vae_checkpoint(path, model: BodyVAE) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "ema_state": {
                "shadow_params": [
                    parameter.detach().clone() for parameter in model.parameters()
                ]
            },
        },
        path,
    )


def _make_config(tmp_path, *, chunk_size=1, noise_steps=1):
    motion_stats, latent_stats = write_statistics(
        tmp_path, latent_dim=8, latent_mean=2.0, latent_std=3.0
    )
    vae = BodyVAE(
        motion_stats_path=motion_stats,
        latent_stats_path=latent_stats,
        latent_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=1,
    )
    checkpoint = tmp_path / "vae.ckpt"
    _write_vae_checkpoint(checkpoint, vae)
    root_stats = tmp_path / "root_stats.npz"
    np.savez(
        root_stats,
        root_mean=np.array([1.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float32),
        root_std=np.array([2.0, 2.0, 2.0, 0.5, 0.5], dtype=np.float32),
    )
    text_embeddings = tmp_path / "text_embeddings.pt"
    embedding_values = {
        "": torch.zeros(2, 8),
        "walk": torch.ones(2, 8),
        "turn": torch.full((2, 8), 2.0),
        "sit": torch.full((2, 8), 3.0),
    }
    torch.save(
        {
            "embeddings": embedding_values,
            "text_dim": 8,
            "text_len": 8,
            "content_id": create_text_embedding_content_id(
                embedding_values, text_dim=8, text_len=8
            ),
        },
        text_embeddings,
    )
    return OmegaConf.create(
        {
            "seed": 7,
            "trainer": {"max_steps": 10},
            "validation": {
                "seed": 11,
                "continuation_history_tokens": 1,
                "self_forcing_steps": 2,
                "self_forcing_history_tokens": 1,
            },
            "training": {"text_dropout_probability": 0.0},
            "loss": {"root_weight": 1.0, "body_weight": 1.0},
            "self_forcing": {
                "enabled": False,
                "phase_start_step": 10,
                "phase_steps": 20,
            },
            "root_stats_path": str(root_stats),
            "text_embeddings_path": str(text_embeddings),
            "vae": {
                "target": "models.vae_wan_1d.BodyVAE",
                "checkpoint_path": str(checkpoint),
                "params": {
                    "motion_stats_path": str(motion_stats),
                    "latent_stats_path": str(latent_stats),
                    "latent_dim": 8,
                    "hidden_dim": 16,
                    "encoder_layers": 1,
                    "decoder_layers": 1,
                    "kernel_size": 3,
                    "dropout": 0.0,
                    "fps": 20.0,
                },
            },
            "model": {
                "target": "models.diffusion_forcing_wan.LDF",
                "ema_decay": 0.99,
                "params": {
                    "hidden_dim": 16,
                    "ffn_dim": 32,
                    "freq_dim": 8,
                    "text_dim": 8,
                    "text_len": 8,
                    "num_heads": 2,
                    "root_num_layers": 1,
                    "body_num_layers": 1,
                    "chunk_size": chunk_size,
                    "noise_steps": noise_steps,
                    "fps": 20.0,
                },
            },
        }
    )


def test_ldf_training_bridge_uses_frozen_ema_vae_and_shared_statistics(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path))
    assert module.model.latent_dim == module.vae.latent_dim == 8
    assert module.vae.encoder_context_tokens == 4
    assert torch.equal(module.model.local_root_mean, module.vae.local_root_mean)
    assert torch.equal(module.model.local_root_std, module.vae.local_root_std)
    assert not any(parameter.requires_grad for parameter in module.vae.parameters())
    module.train()
    assert not module.vae.training


def test_create_clean_motion_aligns_root_latent_and_padding(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path)).eval()
    body_with_context = torch.randn(2, 8, 265)
    encoder_frame_valid = torch.ones(2, 8, dtype=torch.bool)
    context_count = torch.tensor([0, 1], dtype=torch.long)
    root = torch.zeros(2, 8, 5)
    root[..., 3] = 1.0
    root[0, :, 0] = torch.arange(8)
    root[1, :4, 0] = torch.arange(4)
    frame_valid = torch.tensor(
        [
            [True] * 8,
            [True] * 4 + [False] * 4,
        ]
    )
    batch = {
        "root_motion": root,
        "frame_valid_mask": frame_valid,
        "body_with_context": body_with_context,
        "body_with_context_frame_valid_mask": encoder_frame_valid,
        "context_token_count": context_count,
    }

    expected_latent = module.vae.tokenize_window(
        body_with_context, encoder_frame_valid, context_count
    )
    motion, token_valid = module._create_clean_motion(batch)

    assert token_valid.tolist() == [[True, True], [True, False]]
    assert motion.root_motion.shape == (2, 2, 4, 5)
    assert motion.latent_motion.shape == (2, 2, 8)
    assert torch.equal(motion.latent_motion, expected_latent)
    expected_first_root = (
        root[0, :4] - module.model.root_mean
    ) / module.model.root_std
    assert torch.allclose(motion.root_motion[0, 0], expected_first_root)
    assert not motion.root_motion[1, 1].any()
    assert not motion.latent_motion[1, 1].any()
    assert not motion.root_motion.requires_grad
    assert not motion.latent_motion.requires_grad


def test_create_clean_motion_rejects_partial_active_token(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path)).eval()
    batch = {
        "root_motion": torch.zeros(1, 4, 5),
        "frame_valid_mask": torch.tensor([[True, True, False, False]]),
        "body_with_context": torch.zeros(1, 4, 265),
        "body_with_context_frame_valid_mask": torch.ones(1, 4, dtype=torch.bool),
        "context_token_count": torch.zeros(1, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="constant within each four-frame token"):
        module._create_clean_motion(batch)


def test_create_clean_motion_rejects_per_sample_root_body_misalignment(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path)).eval()
    batch = {
        "root_motion": torch.zeros(2, 8, 5),
        "frame_valid_mask": torch.tensor(
            [
                [True] * 8,
                [True] * 4 + [False] * 4,
            ]
        ),
        "body_with_context": torch.zeros(2, 8, 265),
        "body_with_context_frame_valid_mask": torch.tensor(
            [
                [True] * 4 + [False] * 4,
                [True] * 8,
            ]
        ),
        "context_token_count": torch.zeros(2, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="active token counts"):
        module._create_clean_motion(batch)


def test_prompt_timeline_encoding_preserves_token_order_and_builds_null_branch(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path)).eval()
    contexts, null = module._encode_prompt_timeline(
        [["walk", "walk", "turn"], ["sit", "turn", "sit"]],
        apply_dropout=False,
    )

    assert len(contexts) == 6
    assert len(null) == 2
    assert torch.equal(contexts[0], contexts[1])
    assert torch.equal(contexts[2], contexts[4])
    assert not torch.equal(contexts[0], contexts[2])
    assert torch.equal(null[0], null[1])


def test_complete_ldf_training_step_runs_with_frozen_vae_and_text_lookup(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path)).train()
    root = torch.zeros(1, 8, 5)
    root[..., 3] = 1.0
    batch = {
        "root_motion": root,
        "body_motion": torch.randn(1, 8, 265),
        "frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "body_with_context": torch.randn(1, 8, 265),
        "body_with_context_frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "context_token_count": torch.zeros(1, dtype=torch.long),
        "previous_root_frame": torch.zeros(1, 5),
        "previous_root_valid_mask": torch.zeros(1, dtype=torch.bool),
        "source_start_token": torch.zeros(1, dtype=torch.long),
        "cold_start_mask": torch.ones(1, dtype=torch.bool),
        "prompt_timeline": [["walk", "walk"]],
    }
    losses = module._step(batch, is_training=True)
    assert set(losses) == {
        "anchor_root_flow_v",
        "latent_body_flow_v",
        "total",
    }
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in module.model.parameters())
    assert not any(parameter.grad is not None for parameter in module.vae.parameters())


def test_validation_plan_and_noise_are_repeatable_with_fixed_generator(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path)).eval()
    root = torch.zeros(1, 8, 5)
    root[..., 3] = 1.0
    batch = {
        "root_motion": root,
        "body_motion": torch.randn(1, 8, 265),
        "frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "body_with_context": torch.randn(1, 8, 265),
        "body_with_context_frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "context_token_count": torch.zeros(1, dtype=torch.long),
        "previous_root_frame": torch.zeros(1, 5),
        "previous_root_valid_mask": torch.zeros(1, dtype=torch.bool),
        "source_start_token": torch.zeros(1, dtype=torch.long),
        "cold_start_mask": torch.ones(1, dtype=torch.bool),
        "prompt_timeline": [["walk", "walk"]],
    }
    first = module._step(
        batch,
        is_training=False,
        generator=torch.Generator().manual_seed(41),
        rollout_steps_override=1,
        initial_history_tokens=0,
    )
    second = module._step(
        batch,
        is_training=False,
        generator=torch.Generator().manual_seed(41),
        rollout_steps_override=1,
        initial_history_tokens=0,
    )
    assert all(torch.equal(first[name], second[name]) for name in first)


def test_ldf_resume_rejects_statistics_before_overwriting_model(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path / "source"))
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    resumed = LDFLightningModule(_make_config(tmp_path / "source"))
    resumed.on_load_checkpoint(checkpoint)
    original = resumed.model.root_mean.clone()

    bad_checkpoint = dict(checkpoint)
    bad_checkpoint["state_dict"] = dict(checkpoint["state_dict"])
    bad_checkpoint["state_dict"]["root_mean"] = original + 1.0
    with pytest.raises(RuntimeError, match="statistics mismatch for root_mean"):
        resumed.on_load_checkpoint(bad_checkpoint)
    assert torch.equal(resumed.model.root_mean, original)


def test_ldf_resume_rejects_changed_vae_statistics_contract(tmp_path):
    module = LDFLightningModule(_make_config(tmp_path))
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)
    checkpoint["ldf_training_contract"]["vae_statistics"]["latent_mean"] += 1

    resumed = LDFLightningModule(_make_config(tmp_path))
    with pytest.raises(RuntimeError, match="VAE statistics mismatch for latent_mean"):
        resumed.on_load_checkpoint(checkpoint)


def test_ldf_resume_rejects_changed_text_embedding_at_the_same_path(tmp_path):
    cfg = _make_config(tmp_path)
    module = LDFLightningModule(cfg)
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    path = tmp_path / "text_embeddings.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["embeddings"]["walk"] = torch.full((2, 8), 9.0)
    payload["content_id"] = create_text_embedding_content_id(
        payload["embeddings"], text_dim=8, text_len=8
    )
    torch.save(payload, path)

    resumed = LDFLightningModule(cfg)
    with pytest.raises(RuntimeError, match="text embedding content"):
        resumed.on_load_checkpoint(checkpoint)


def test_real_dataset_vae_and_self_forcing_kernel_backpropagate_only_final_step(
    tmp_path,
):
    data_root = tmp_path / "dataset"
    artifacts = data_root / "artifacts"
    artifacts.mkdir(parents=True)
    (data_root / "train.txt").write_text("sample\n")
    root = np.zeros((56, 5), dtype=np.float32)
    root[:, 0] = np.arange(56, dtype=np.float32)
    root[:, 2] = np.arange(56, dtype=np.float32) * 0.5
    root[:, 3] = 1.0
    body = np.random.default_rng(3).standard_normal((56, 265)).astype(np.float32)
    np.savez(
        artifacts / "sample.npz",
        root_motion=root,
        body_motion=body,
        body_feature_valid_mask=np.ones_like(body, dtype=np.bool_),
    )
    dataset = HumanML3DDataset(
        meta_paths=[data_root / "train.txt"],
        split="train",
        artifact_path="artifacts",
        text_path=None,
    )
    module = LDFLightningModule(
        _make_config(tmp_path / "model", chunk_size=5, noise_steps=10)
    ).train()
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        encoder_context_tokens=module.vae.encoder_context_tokens,
        training=False,
        cold_start=False,
    )([dataset[0]])
    plan = sample_window_plan(
        batch,
        active_tokens=module.model.chunk_size,
        rollout_steps=2,
        latent_dim=module.model.latent_dim,
        initial_history_tokens=2,
        phase_offset=torch.tensor([0.05]),
        generator=torch.Generator().manual_seed(9),
    )
    anchored = anchor_physical_batch(batch, plan.translation_anchor_xz)
    clean_motion, token_valid = module._create_clean_motion(anchored)
    assert token_valid.all()

    def condition_builder(_view):
        return LDFCondition(
            text_context=[torch.zeros(1, 8)],
            text_null_context=[torch.zeros(1, 8)],
        )

    result = run_self_forcing_rollout(
        module.model,
        SelfForcingState(clean_motion),
        plan,
        previous_root_frame=anchored["previous_root_frame"],
        previous_root_valid_mask=anchored["previous_root_valid_mask"],
        condition_builder=condition_builder,
    )
    losses = compute_velocity_loss(result.prediction, result.final_step)
    losses["total"].backward()

    assert all(parameter.grad is None for parameter in module.vae.parameters())
    assert any(
        parameter.grad is not None
        for parameter in module.model.root_transformer.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in module.model.body_transformer.parameters()
    )
    assert len(result.replacements) == 1
    assert result.replacements[0].root_motion.grad_fn is None
    assert result.replacements[0].latent_motion.grad_fn is None
    assert result.final_step.loss_mask.sum().item() == 5
