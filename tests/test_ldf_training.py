from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from datasets.humanml3d import HumanML3DDataset
from models.vae_wan_1d import BodyVAE
from tests.vae_helpers import write_statistics
from utils.conditions.ldf import LDFCondition
from utils.training.ldf.batch import LDFStepView, anchor_physical_batch
from utils.training.ldf.conditioning import (
    create_xz_condition,
    sample_constraint_keep_mask,
    sample_xz_constraint_mask,
)
from utils.training.ldf.losses import compute_velocity_loss
from utils.training.ldf.data import LDFSpanCollator
from utils.training.ldf.lightning_module import (
    LDFLightningModule,
    _create_curriculum_generator,
)
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
            },
            "training": {
                "text_dropout": 0.0,
                "constraint_dropout": 0.0,
                "max_horizon_token": 2,
                "window": {
                    "max_tokens": 50,
                    "generation_tokens": chunk_size,
                    "sampling": "random_generation_start",
                },
                "constraint_sampling": {
                    "dense_probability": 1.0,
                    "waypoint_probability": 0.0,
                    "goal_probability": 0.0,
                    "max_waypoint_count": 4,
                },
            },
            "loss": {"root_weight": 1.0, "body_weight": 1.0},
            "self_forcing": {
                "enabled": False,
                "phase_start_step": 10,
                "phase_steps": 20,
                "k_schedule": [[0.0, 2], [0.4, 3], [0.7, 5]],
                "teacher_replay": {2: 0.2, 3: 0.1, 5: 0.1},
            },
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
                },
            },
            "data": {
                "root_stats_path": str(root_stats),
                "text_embeddings_path": str(text_embeddings),
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


def _make_step_batch():
    root = torch.zeros(1, 8, 5)
    root[..., 3] = 1.0
    return {
        "root_motion": root,
        "body_motion": torch.randn(1, 8, 265),
        "frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "body_with_context": torch.randn(1, 8, 265),
        "body_with_context_frame_valid_mask": torch.ones(1, 8, dtype=torch.bool),
        "context_token_count": torch.zeros(1, dtype=torch.long),
        "previous_root_frame": torch.zeros(1, 5),
        "previous_root_valid_mask": torch.zeros(1, dtype=torch.bool),
        "source_start_token": torch.zeros(1, dtype=torch.long),
        "span_token_count": torch.tensor([2]),
        "prompt_timeline": [["walk", "walk"]],
    }


def test_step_keeps_teacher_forcing_before_phase_start(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    cfg.self_forcing.enabled = True
    module = LDFLightningModule(cfg).train()

    def fail_if_sampled(*_args, **_kwargs):
        pytest.fail("self-forcing sampling must not run before phase_start_step")

    monkeypatch.setattr(
        "utils.training.ldf.lightning_module.sample_rollout_steps",
        fail_if_sampled,
    )
    losses = module._step(
        _make_step_batch(),
        is_training=True,
        initial_history_tokens=0,
    )
    assert torch.isfinite(losses["total"])


def test_step_starts_self_forcing_at_phase_boundary(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path)
    cfg.self_forcing.enabled = True
    cfg.self_forcing.phase_start_step = 0
    module = LDFLightningModule(cfg).train()
    calls = []

    def record_sampling(progress, **_kwargs):
        calls.append(progress)
        return 1

    monkeypatch.setattr(
        "utils.training.ldf.lightning_module.sample_rollout_steps",
        record_sampling,
    )
    module._step(
        _make_step_batch(),
        is_training=True,
        initial_history_tokens=0,
    )
    assert calls == [0.0]


def test_curriculum_generator_is_global_step_deterministic_and_rank_independent():
    first = _create_curriculum_generator(1234, 200_000)
    second = _create_curriculum_generator(1234, 200_000)
    assert first.device == second.device == torch.device("cpu")
    assert torch.equal(torch.rand(8, generator=first), torch.rand(8, generator=second))


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


def test_xz_condition_exposes_only_active_xz_and_post_active_lookahead():
    root = torch.arange(2 * 8 * 4 * 5, dtype=torch.float32).reshape(2, 8, 4, 5)
    token_valid = torch.tensor(
        [[True] * 8, [True] * 7 + [False]], dtype=torch.bool
    )
    positions = torch.arange(10, 18)[None].expand(2, -1)
    view = LDFStepView(
        step_index=0,
        history_end=torch.tensor([2, 2]),
        active_start=torch.tensor([2, 2]),
        active_end=torch.tensor([5, 5]),
        frontier_start=torch.tensor([5, 5]),
        timeline_position_ids=positions,
        rope_position_ids=torch.arange(-2, 6)[None].expand(2, -1),
        beta=torch.zeros(2, 8),
    )
    text = [torch.ones(1, 8) for _ in range(16)]
    null = [torch.zeros(1, 8) for _ in range(2)]
    constraint_mask = sample_xz_constraint_mask(
        token_valid_mask=token_valid,
        initial_active_start=torch.tensor([2, 2]),
        initial_active_end=torch.tensor([5, 5]),
        max_horizon_token=2,
        dense_probability=1.0,
        waypoint_probability=0.0,
        goal_probability=0.0,
        max_waypoint_count=4,
    )
    constraint_mask[1] = False
    condition = create_xz_condition(
        clean_root_motion=root,
        token_valid_mask=token_valid,
        constraint_mask=constraint_mask,
        view=view,
        text_context=text,
        text_null_context=null,
        max_horizon_token=2,
    )

    mask = condition.root_condition_mask
    assert mask[0, 2:5, :, 0].all() and mask[0, 2:5, :, 2].all()
    assert not mask[0, :, :, 1].any()
    assert not mask[0, :, :, 3:].any()
    assert not mask[0, :2].any() and not mask[0, 5:].any()
    assert not mask[1].any()
    assert torch.equal(condition.root_condition_value, root)
    assert condition.future_root_condition_value.shape == (2, 2, 4, 5)
    assert condition.future_timeline_position_ids.tolist() == [[15, 16], [0, 0]]
    assert condition.future_valid_mask.tolist() == [[True, True], [False, False]]
    assert condition.future_root_condition_mask[0, :, :, 0].all()
    assert condition.future_root_condition_mask[0, :, :, 2].all()
    assert not condition.future_root_condition_mask[..., 1].any()
    assert not condition.future_root_condition_mask[..., 3:].any()


def test_xz_sampling_supports_dense_waypoints_and_single_future_goal():
    valid = torch.ones(1, 8, dtype=torch.bool)
    common = dict(
        token_valid_mask=valid,
        initial_active_start=torch.tensor([2]),
        initial_active_end=torch.tensor([5]),
        max_horizon_token=2,
        max_waypoint_count=4,
    )
    dense = sample_xz_constraint_mask(
        **common,
        dense_probability=1.0,
        waypoint_probability=0.0,
        goal_probability=0.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert dense[:, 2:7, :, 0].all()
    assert dense[:, 2:7, :, 2].all()
    assert not dense[:, 7:].any()
    assert not dense[:, :2].any()

    waypoints = sample_xz_constraint_mask(
        **common,
        dense_probability=0.0,
        waypoint_probability=1.0,
        goal_probability=0.0,
        generator=torch.Generator().manual_seed(2),
    )
    selected_waypoint_frames = int(waypoints[..., 0].sum().item())
    assert 1 <= selected_waypoint_frames <= 4
    assert not waypoints[:, :2].any()
    assert not waypoints[:, 7:].any()

    goal = sample_xz_constraint_mask(
        **common,
        dense_probability=0.0,
        waypoint_probability=0.0,
        goal_probability=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert not goal[:, :5].any()
    assert int(goal[..., 0].sum().item()) == 1
    assert int(goal[:, 5:7, :, 0].sum().item()) == 1
    for mask in (dense, waypoints, goal):
        assert torch.equal(mask[..., 0], mask[..., 2])
        assert not mask[..., 1].any()
        assert not mask[..., 3:].any()


def test_goal_sampling_without_future_falls_back_to_one_active_waypoint():
    goal = sample_xz_constraint_mask(
        token_valid_mask=torch.ones(1, 5, dtype=torch.bool),
        initial_active_start=torch.tensor([2]),
        initial_active_end=torch.tensor([5]),
        max_horizon_token=2,
        dense_probability=0.0,
        waypoint_probability=0.0,
        goal_probability=1.0,
        max_waypoint_count=4,
        generator=torch.Generator().manual_seed(4),
    )
    assert int(goal[..., 0].sum().item()) == 1
    assert int(goal[:, 2:5, :, 0].sum().item()) == 1


def test_sparse_future_constraints_are_packed_by_absolute_position():
    root = torch.randn(2, 8, 4, 5)
    valid = torch.ones(2, 8, dtype=torch.bool)
    mask = torch.zeros_like(root, dtype=torch.bool)
    mask[0, 5, 1, 0] = mask[0, 5, 1, 2] = True
    mask[0, 7, 3, 0] = mask[0, 7, 3, 2] = True
    mask[1, 6, 0, 0] = mask[1, 6, 0, 2] = True
    positions = torch.arange(10, 18)[None].expand(2, -1)
    view = LDFStepView(
        step_index=0,
        history_end=torch.tensor([2, 2]),
        active_start=torch.tensor([2, 2]),
        active_end=torch.tensor([5, 5]),
        frontier_start=torch.tensor([5, 5]),
        timeline_position_ids=positions,
        rope_position_ids=torch.arange(-2, 6)[None].expand(2, -1),
        beta=torch.zeros(2, 8),
    )
    condition = create_xz_condition(
        clean_root_motion=root,
        token_valid_mask=valid,
        constraint_mask=mask,
        view=view,
        text_context=[torch.zeros(1, 8) for _ in range(16)],
        text_null_context=[torch.zeros(1, 8) for _ in range(2)],
        max_horizon_token=3,
    )
    assert condition.future_timeline_position_ids.tolist() == [[15, 17], [16, 0]]
    assert condition.future_valid_mask.tolist() == [[True, True], [True, False]]
    assert condition.future_root_condition_mask[0, 0, 1, 0]
    assert condition.future_root_condition_mask[0, 1, 3, 2]
    assert condition.future_root_condition_mask[1, 0, 0, 0]
    assert not condition.future_root_condition_mask[1, 1].any()


def test_constraint_dropout_is_per_sample_and_independent_of_text_dropout():
    device = torch.device("cpu")
    assert sample_constraint_keep_mask(
        3,
        dropout_probability=0.0,
        device=device,
        apply_dropout=True,
    ).tolist() == [True, True, True]
    assert sample_constraint_keep_mask(
        3,
        dropout_probability=1.0,
        device=device,
        apply_dropout=True,
    ).tolist() == [False, False, False]
    # Validation always preserves constraints, regardless of the train dropout rate.
    assert sample_constraint_keep_mask(
        3,
        dropout_probability=1.0,
        device=device,
        apply_dropout=False,
    ).tolist() == [True, True, True]


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
        "span_token_count": torch.tensor([2]),
        "prompt_timeline": [["walk", "walk"]],
    }
    seen_conditions = []
    original_forward = module.model.forward

    def recording_forward(inputs):
        seen_conditions.append(inputs.condition)
        return original_forward(inputs)

    module.model.forward = recording_forward
    losses = module._step(batch, is_training=True, initial_history_tokens=0)
    assert set(losses) == {
        "anchor_root_flow_v",
        "latent_body_flow_v",
        "anchor_root_offpath_endpoint",
        "latent_body_offpath_endpoint",
        "root_boundary_displacement",
        "total",
    }
    assert torch.isfinite(losses["total"])
    assert len(seen_conditions) == 1
    condition = seen_conditions[0]
    assert condition.root_condition_mask[:, 0, :, 0].all()
    assert condition.root_condition_mask[:, 0, :, 2].all()
    assert not condition.root_condition_mask[:, 0, :, 1].any()
    assert condition.future_valid_mask.tolist() == [[True]]
    assert condition.future_timeline_position_ids.tolist() == [[1]]
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
        "span_token_count": torch.tensor([2]),
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
        generation_tokens=5,
        encoder_context_tokens=module.vae.encoder_context_tokens,
        training=False,
    )([dataset[0]])
    plan = sample_window_plan(
        batch,
        active_tokens=module.model.chunk_size,
        rollout_steps=2,
        latent_dim=module.model.latent_dim,
        initial_history_tokens=2,
        phase_offset=torch.zeros(1),
        generator=torch.Generator().manual_seed(9),
    )
    anchored = anchor_physical_batch(batch, plan.translation_anchor_xz)
    clean_motion, token_valid = module._create_clean_motion(anchored)
    assert token_valid.all()

    def condition_builder(_view, _clean_motion):
        null = torch.zeros(1, 8)
        return LDFCondition(
            text_context=[null for _ in range(clean_motion.token_length)],
            text_null_context=[null],
        )

    result = run_self_forcing_rollout(
        module.model,
        SelfForcingState(clean_motion),
        plan,
        previous_root_frame=anchored["previous_root_frame"],
        previous_root_valid_mask=anchored["previous_root_valid_mask"],
        condition_builder=condition_builder,
    )
    from utils.training.ldf.losses import compute_offpath_loss

    losses = compute_offpath_loss(
        result.prediction,
        result.final_step,
        root_mean=module.model.root_mean,
        root_std=module.model.root_std,
    )
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
    assert len(result.replacements) == 2
    assert result.replacements[0].root_motion.grad_fn is None
    assert result.replacements[0].latent_motion.grad_fn is None
    assert result.final_step.loss_mask.sum().item() == 5
