from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from torch.multiprocessing.spawn import ProcessRaisedException

from models.diffusion_forcing_wan import LDF
from eval.ldf_training import LDFEvaluationCallback
from metrics.trajectory import (
    compute_dense_xz_metrics,
    summarize_dense_xz_records,
)
from tests.vae_helpers import make_vae
from utils.conditions.ldf import HybridMotion, LDFPrediction
from utils.training.ldf.evaluation import artifacts as evaluation_artifacts
from utils.training.ldf.evaluation import runner as evaluation_runner
from utils.training.ldf.evaluation.artifacts import save_dense_xz_sample
from utils.training.ldf.evaluation.generation import (
    compile_evaluation_prompt,
    create_evaluation_initial_noise,
    generate_evaluation_sequence,
    rotate_evaluation_sample,
)
from utils.training.ldf.evaluation.runner import (
    LDFEvaluationRunner,
    _all_gather_objects,
    _compact_rollout_metrics,
    _distributed_barrier,
    _format_t2m_summary,
    _standard_case_metrics,
)


class _TextEmbeddings:
    @staticmethod
    def lookup(texts):
        return [torch.zeros(1, 4) for _ in texts]


def _distributed_collective_worker(rank: int, world_size: int, init_file: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        gathered = _all_gather_objects({"rank": rank, "samples": [rank, rank + 2]})
        assert gathered == [
            {"rank": 0, "samples": [0, 2]},
            {"rank": 1, "samples": [1, 3]},
        ]
        _distributed_barrier()
    finally:
        dist.destroy_process_group()


def _zero_prediction(self, inputs, **kwargs):
    seen_cfg_modes = getattr(self, "_seen_cfg_modes", None)
    if seen_cfg_modes is not None:
        seen_cfg_modes.append(kwargs.get("mode"))
    seen_cfg_scales = getattr(self, "_seen_cfg_scales", None)
    if seen_cfg_scales is not None:
        seen_cfg_scales.append(kwargs.get("cfg_scale_joint"))
    root_velocity = torch.zeros_like(inputs.noisy_motion.root_motion)
    latent_velocity = torch.zeros_like(inputs.noisy_motion.latent_motion)
    local = torch.zeros(*root_velocity.shape[:3], 4, device=root_velocity.device)
    valid = torch.ones_like(local, dtype=torch.bool)
    return LDFPrediction(
        raw_root_output=inputs.noisy_motion.root_motion,
        raw_body_output=latent_velocity,
        clean_motion=inputs.noisy_motion,
        solver_velocity=HybridMotion(root_velocity, latent_velocity),
        local_root_motion=local,
        local_root_feature_valid=valid,
    )


def _evaluation_module():
    model = LDF(
        latent_dim=3,
        local_root_mean=[0] * 4,
        local_root_std=[1] * 4,
        hidden_dim=8,
        ffn_dim=16,
        freq_dim=8,
        text_dim=4,
        text_len=4,
        num_heads=2,
        root_num_layers=1,
        body_num_layers=1,
        chunk_size=2,
        noise_steps=2,
    ).eval()
    model.predict_with_cfg = types.MethodType(_zero_prediction, model)
    vae = make_vae(
        latent_dim=3,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    ).eval()
    return types.SimpleNamespace(
        model=model,
        vae=vae,
        text_embeddings=_TextEmbeddings(),
    )


def test_generation_evaluation_is_composed_as_an_entrypoint_callback():
    callback = object.__new__(LDFEvaluationCallback)
    calls = []
    coverage_calls = []
    startup_calls = []
    callback.runner = types.SimpleNamespace(
        maybe_run=calls.append,
        validate_text_coverage=coverage_calls.append,
        run_at_start=lambda module: startup_calls.append(module) or True,
    )
    module = object()

    callback.on_validation_start(types.SimpleNamespace(is_global_zero=True), module)
    callback.on_validation_start(types.SimpleNamespace(is_global_zero=False), module)
    assert coverage_calls == [module, module]
    callback.on_validation_epoch_end(
        types.SimpleNamespace(sanity_checking=False, is_global_zero=True),
        module,
    )
    assert startup_calls == [module]
    assert calls == []
    callback.runner.run_at_start = lambda module: False
    callback.on_validation_epoch_end(
        types.SimpleNamespace(sanity_checking=False, is_global_zero=True),
        module,
    )
    assert calls == [module]
    callback.on_validation_epoch_end(
        types.SimpleNamespace(sanity_checking=True, is_global_zero=True),
        module,
    )
    callback.on_validation_epoch_end(
        types.SimpleNamespace(sanity_checking=False, is_global_zero=False),
        module,
    )
    assert calls == [module, module]


def test_startup_yaw_video_probe_runs_once_before_optimizer_steps(monkeypatch):
    cfg = OmegaConf.create(
        {
            "validation": {
                "generation": {
                    "enabled": True,
                    "run_at_start": True,
                    "render": True,
                },
                "dense_xz": {"enabled": True},
                "t2m": {"enabled": False},
            },
            "train": True,
        }
    )
    runner = LDFEvaluationRunner(cfg)
    calls = []
    monkeypatch.setattr(runner, "_evaluation_context", lambda module: nullcontext())
    monkeypatch.setattr(
        runner,
        "_run_startup_yaw_videos",
        lambda module, **kwargs: calls.append(kwargs),
    )
    module = types.SimpleNamespace(global_step=160_000)

    assert runner.run_at_start(module) is True
    assert runner.run_at_start(module) is False

    assert calls == [{"step": 160_000, "step_tag": "fit_start"}]


def test_t2m_console_summary_reports_all_computed_metrics():
    output = _format_t2m_summary(
        {
            "num_samples": 1450,
            "cfg_mode": "nocfg",
            "FID": 1.23456,
            "Matching_score": 2.34567,
            "gt_Matching_score": 1.98766,
            "R_precision_top_1": 0.41,
            "R_precision_top_2": 0.62,
            "R_precision_top_3": 0.73,
            "gt_R_precision_top_1": 0.51,
            "gt_R_precision_top_2": 0.72,
            "gt_R_precision_top_3": 0.83,
            "Diversity": 8.76543,
            "gt_Diversity": 9.01234,
        },
        mode="stream",
        step_tag="step_010000",
    )

    assert "[t2m][stream][step_010000] samples=1450 cfg=nocfg" in output
    assert "FID=1.2346" in output
    assert "MatchingScore: generated=2.3457, ground_truth=1.9877" in output
    assert "R-Precision: top1=0.4100, top2=0.6200, top3=0.7300" in output
    assert "GT R-Precision: top1=0.5100, top2=0.7200, top3=0.8300" in output
    assert "Diversity: generated=8.7654, ground_truth=9.0123" in output


def test_generation_evaluation_shards_samples_without_loading_peer_motion():
    class MetadataDataset:
        def __init__(self):
            self.dataset = [
                {"dataset": "HumanML3D", "name": f"sample_{index}"}
                for index in range(7)
            ]
            self.loaded = []

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            self.loaded.append(index)
            return dict(self.dataset[index])

    cfg = OmegaConf.create({"validation": {"generation": {"enabled": False}}})
    assignments = []
    for rank in range(3):
        dataset = MetadataDataset()
        runner = LDFEvaluationRunner(cfg)
        runner._dataset = dataset
        shard, total = runner._selected_sample_shard(
            types.SimpleNamespace(global_rank=rank, world_size=3),
            limit=0,
            dataset_name="HumanML3D",
        )
        indices = [index for index, _ in shard]
        assert total == 7
        assert dataset.loaded == indices
        assignments.extend(indices)
    assert sorted(assignments) == list(range(7))


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_generation_evaluation_collectives_run_in_two_real_processes(tmp_path):
    world_size = 2
    try:
        mp.spawn(
            _distributed_collective_worker,
            args=(world_size, str(tmp_path / "ddp_init")),
            nprocs=world_size,
            join=True,
        )
    except ProcessRaisedException as error:
        message = str(error)
        if "Operation not permitted" in message or "Cannot resolve" in message:
            pytest.skip("sandbox does not permit the Gloo loopback transport")
        raise


def test_evaluation_text_coverage_fails_before_training_for_missing_probe_prompt():
    cfg = OmegaConf.create(
        {
            "validation": {
                "generation": {"enabled": True},
                "dense_xz": {"enabled": True, "probe": "dense_xz"},
                "t2m": {"enabled": False},
            },
            "data": {
                "max_frames": 8,
                "test_probe_meta_paths": {"dense_xz": ["unused.txt"]},
            },
        }
    )
    runner = LDFEvaluationRunner(cfg)
    runner._probe_datasets["dense_xz"] = [
        {
            "dataset": "HumanML3D",
            "name": "probe",
            "root_motion": torch.zeros(8, 5),
            "text_data": [
                {
                    "text": "missing prompt",
                    "tokens": [],
                    "start_frame": 0,
                    "end_frame": 8,
                }
            ],
        }
    ]

    class MissingLookup:
        @staticmethod
        def lookup(texts):
            raise KeyError(texts)

    with pytest.raises(RuntimeError, match="EVALUATION_TEXT_EMBEDDINGS_INCOMPLETE"):
        runner.validate_text_coverage(
            types.SimpleNamespace(text_embeddings=MissingLookup())
        )


def test_dense_xz_metrics_keep_only_time_aligned_control_errors():
    target = torch.zeros(8, 5)
    target[:, 0] = torch.arange(8) * 0.1
    predicted = target.clone()
    predicted[:, 2] += 0.1
    metrics = compute_dense_xz_metrics(
        predicted,
        target,
    )
    assert metrics["ade"] == pytest.approx(0.1)
    assert metrics["fde"] == pytest.approx(0.1)
    assert metrics["max_error"] == pytest.approx(0.1)
    assert set(metrics) == {"frames", "ade", "fde", "max_error"}


def test_dense_xz_summary_prioritizes_control_and_heading_metrics():
    summary = summarize_dense_xz_records(
        [
            {
                "ade": 0.1,
                "fde": 0.2,
                "max_error": 0.3,
                "root_gt_root_heading_angle_deg": 10.0,
                "body_gt_body_heading_angle_deg": 15.0,
                "feet_gt_feet_heading_angle_deg": 17.0,
                "root_trajectory_heading_angle_deg": 18.0,
                "root_body_heading_angle_deg": 20.0,
                "root_feet_heading_angle_deg": 25.0,
            },
            {
                "ade": 0.3,
                "fde": 0.4,
                "max_error": 0.5,
                "root_gt_root_heading_angle_deg": 30.0,
                "body_gt_body_heading_angle_deg": 35.0,
                "feet_gt_feet_heading_angle_deg": 37.0,
                "root_trajectory_heading_angle_deg": 38.0,
                "root_body_heading_angle_deg": 40.0,
                "root_feet_heading_angle_deg": 45.0,
            },
        ]
    )
    assert summary["num_samples"] == 2
    assert summary["ade_mean"] == pytest.approx(0.2)
    assert summary["fde_std"] == pytest.approx(0.1)
    assert summary["max_error_mean"] == pytest.approx(0.4)
    assert summary["root_gt_root_heading_angle_deg_mean"] == pytest.approx(20.0)
    assert summary["body_gt_body_heading_angle_deg_mean"] == pytest.approx(25.0)
    assert summary["feet_gt_feet_heading_angle_deg_mean"] == pytest.approx(27.0)
    assert summary["root_trajectory_heading_angle_deg_mean"] == pytest.approx(28.0)
    assert summary["root_body_heading_angle_deg_mean"] == pytest.approx(30.0)
    assert summary["root_feet_heading_angle_deg_mean"] == pytest.approx(35.0)


def test_compact_rollout_and_standard_case_logs_use_short_names():
    records = [
        {
            "name": "000021",
            "cold_root_deg": 10.0,
            "cold_root_max": 20.0,
            "cold_root_anti": 0.0,
            "cold_body_deg": 11.0,
            "cold_feet_deg": 12.0,
            "roll_root_deg": 13.0,
            "roll_root_p95": 14.0,
            "roll_root_max": 30.0,
            "roll_root_anti": 0.1,
            "roll_body_deg": 15.0,
            "roll_feet_deg": 16.0,
            "roll_body_rel": 4.0,
            "roll_body_rel_max": 8.0,
            "roll_feet_rel": 5.0,
            "roll_feet_rel_max": 9.0,
            "roll_feet_rev": 0.2,
            "ade": 0.3,
            "fde": 0.4,
        },
        {
            "name": "000021",
            "cold_root_deg": 30.0,
            "cold_root_max": 40.0,
            "cold_root_anti": 1.0,
            "cold_body_deg": 31.0,
            "cold_feet_deg": 32.0,
            "roll_root_deg": 33.0,
            "roll_root_p95": 34.0,
            "roll_root_max": 50.0,
            "roll_root_anti": 0.3,
            "roll_body_deg": 35.0,
            "roll_feet_deg": 36.0,
            "roll_body_rel": 6.0,
            "roll_body_rel_max": 10.0,
            "roll_feet_rel": 7.0,
            "roll_feet_rel_max": 11.0,
            "roll_feet_rev": 0.4,
            "ade": 0.5,
            "fde": 0.6,
        },
    ]

    compact = _compact_rollout_metrics(records)
    assert compact["val/cold/root_deg"] == pytest.approx(20.0)
    assert compact["val/roll/root_p95"] == pytest.approx(24.0)
    assert compact["val/roll/feet_rev"] == pytest.approx(0.3)
    assert compact["val/roll/ade"] == pytest.approx(0.4)
    assert set(compact) == {
        "val/cold/root_deg",
        "val/cold/root_anti",
        "val/cold/body_deg",
        "val/cold/feet_deg",
        "val/roll/root_deg",
        "val/roll/root_p95",
        "val/roll/root_anti",
        "val/roll/body_deg",
        "val/roll/feet_deg",
        "val/roll/body_rel",
        "val/roll/feet_rel",
        "val/roll/feet_rev",
        "val/roll/ade",
        "val/roll/fde",
    }

    case = _standard_case_metrics(records, case_name="000021")
    assert case["val/case/000021/cold_mean"] == pytest.approx(20.0)
    assert case["val/case/000021/cold_max"] == pytest.approx(40.0)
    assert case["val/case/000021/root_mean"] == pytest.approx(23.0)
    assert case["val/case/000021/root_max"] == pytest.approx(50.0)
    assert case["val/case/000021/body_rel_mean"] == pytest.approx(5.0)
    assert case["val/case/000021/body_rel_max"] == pytest.approx(10.0)
    assert case["val/case/000021/feet_rel_mean"] == pytest.approx(6.0)
    assert case["val/case/000021/feet_rel_max"] == pytest.approx(11.0)


def test_evaluation_prompt_is_deterministic_for_humanml_and_babel():
    human = {
        "dataset": "HumanML3D",
        "text_data": [
            {"text": "walk", "tokens": ["walk/VERB"], "start_frame": 0, "end_frame": 8},
            {"text": "move", "tokens": ["move/VERB"], "start_frame": 0, "end_frame": 8},
        ],
    }
    prompt = compile_evaluation_prompt(human, frame_count=8)
    assert prompt.timeline == ("walk", "walk")
    assert prompt.tokens == ("walk/VERB",)

    babel = {
        "dataset": "BABEL",
        "text_data": [
            {"text": "walk", "tokens": [], "start_frame": 0, "end_frame": 5},
            {"text": "turn", "tokens": [], "start_frame": 5, "end_frame": 12},
        ],
    }
    prompt = compile_evaluation_prompt(babel, frame_count=12)
    assert prompt.timeline == ("walk", "turn", "turn")
    assert prompt.change_frames.tolist() == [0, 4, 12]


def test_paired_yaw_source_keeps_latent_noise_and_rotates_root_noise():
    module = _evaluation_module()
    base = create_evaluation_initial_noise(
        module,
        window_tokens=4,
        seed=17,
        yaw_degrees=0,
    )
    rotated = create_evaluation_initial_noise(
        module,
        window_tokens=4,
        seed=17,
        yaw_degrees=90,
    )
    assert torch.equal(rotated.latent_motion, base.latent_motion)
    assert torch.allclose(rotated.root_motion[..., 0], base.root_motion[..., 2])
    assert torch.allclose(rotated.root_motion[..., 2], -base.root_motion[..., 0])
    assert torch.allclose(rotated.root_motion[..., 3], -base.root_motion[..., 4])
    assert torch.allclose(rotated.root_motion[..., 4], base.root_motion[..., 3])


def test_dense_xz_yaw_video_probe_renders_only_configured_rotation_triplet(
    tmp_path,
    monkeypatch,
):
    root = torch.zeros(8, 5)
    root[:, 0] = torch.linspace(0.0, 0.7, 8)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    body = torch.zeros(8, 259)
    sample = {
        "dataset": "HumanML3D",
        "name": "001168",
        "root_motion": root,
        "body_motion": body,
        "text_data": [],
    }
    prompt = types.SimpleNamespace(caption="walk")
    base_generated = types.SimpleNamespace(
        root_motion=root,
        body_motion=body,
        prompt=prompt,
    )
    generation_calls = []
    render_calls = []

    def fake_generate(module, rotated_sample, **kwargs):
        generation_calls.append((rotated_sample, kwargs))
        return types.SimpleNamespace(
            root_motion=rotated_sample["root_motion"],
            body_motion=rotated_sample["body_motion"],
            prompt=prompt,
        )

    monkeypatch.setattr(evaluation_runner, "generate_evaluation_sequence", fake_generate)
    monkeypatch.setattr(
        evaluation_runner,
        "render_comparison_video",
        lambda **kwargs: render_calls.append(kwargs),
    )
    runner = LDFEvaluationRunner(OmegaConf.create({}))
    paths = runner._render_yaw_videos(
        types.SimpleNamespace(model=types.SimpleNamespace(fps=20.0)),
        sample=sample,
        base_generated=base_generated,
        base_target_root=root,
        base_target_body=body,
        frame_count=8,
        seed=4321,
        mode="stream",
        config={
            "rolling_window_tokens": 50,
            "max_horizon_token": 10,
            "num_denoise_steps": 10,
        },
        yaw_degrees=(0.0, 90.0, 180.0),
        video_dir=tmp_path / "video",
        composite_dir=tmp_path / "composite",
    )
    assert [call[1]["initial_noise_yaw_degrees"] for call in generation_calls] == [
        90.0,
        180.0,
    ]
    assert len(render_calls) == 3
    assert [Path(path).name for path in paths] == [
        "001168_yaw_000deg.mp4",
        "001168_yaw_090deg.mp4",
        "001168_yaw_180deg.mp4",
    ]
    assert torch.allclose(
        generation_calls[0][0]["root_motion"][:, 0],
        root[:, 2],
        atol=1e-6,
    )
    assert torch.allclose(
        generation_calls[0][0]["root_motion"][:, 2],
        -root[:, 0],
        atol=1e-6,
    )


def test_dense_xz_yaw_video_config_uses_probe_samples():
    dense = OmegaConf.create(
        {
            "video_yaw_degrees": [0, 90, 180],
        }
    )
    yaw_degrees = LDFEvaluationRunner._yaw_video_config(dense)
    assert yaw_degrees == (0.0, 90.0, 180.0)


def test_dense_xz_artifacts_follow_floodnet_layout(tmp_path):
    root = torch.zeros(8, 5)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    body = torch.zeros(8, 259)
    dirs = save_dense_xz_sample(
        save_dir=tmp_path,
        dataset="HumanML3D",
        probe="dense_xz_stream",
        step_tag="step_010000",
        sample_id="sample",
        caption="walk",
        root_motion=torch.zeros(2, 4, 5),
        latent_motion=torch.zeros(2, 128),
        predicted_root=root,
        predicted_body=body,
        target_root=root,
        target_body=body,
        trajectory_mask=torch.ones(8, dtype=torch.bool),
        prompt_change_frames=np.asarray([0, 8]),
        record={"ade": 0.0, "invalid": float("nan")},
        render=False,
        fps=20.0,
    )
    expected = tmp_path / "HumanML3D" / "metrics" / "dense_xz_stream" / "step_010000"
    assert dirs["metrics"] == expected
    assert (expected / "sample.json").is_file()
    assert (tmp_path / "HumanML3D" / "feature" / "dense_xz_stream" / "step_010000" / "sample.npz").is_file()
    assert json.loads((expected / "sample.json").read_text())["invalid"] is None


def test_dense_xz_video_embeds_trajectory_in_fixed_camera_scene(
    tmp_path,
    monkeypatch,
):
    frames = [
        np.full((128, 128, 3), fill_value=value, dtype=np.uint8)
        for value in (32, 64)
    ]
    written: dict[str, list[np.ndarray]] = {}
    render_calls = []

    class Reader:
        def __iter__(self):
            return iter(frames)

        def close(self):
            return None

    class Writer:
        def __init__(self, path):
            self.path = str(path)
            written[self.path] = []

        def append_data(self, frame):
            written[self.path].append(np.asarray(frame).copy())

        def close(self):
            return None

    def record_render(*args, **kwargs):
        render_calls.append((args, kwargs))

    monkeypatch.setattr(evaluation_artifacts, "render_motion_video", record_render)
    monkeypatch.setattr(
        evaluation_artifacts.imageio,
        "get_reader",
        lambda path: Reader(),
    )
    monkeypatch.setattr(
        evaluation_artifacts.imageio,
        "get_writer",
        lambda path, fps: Writer(path),
    )

    target_root = torch.zeros(2, 5)
    target_root[:, 0] = torch.tensor([0.0, 1.0])
    predicted_root = torch.zeros(2, 5)
    predicted_root[:, 2] = torch.tensor([0.0, 1.0])
    body = torch.zeros(2, 259)
    video_path = tmp_path / "video.mp4"
    composite_path = tmp_path / "composite.mp4"
    evaluation_artifacts.render_comparison_video(
        target_root=target_root,
        target_body=body,
        predicted_root=predicted_root,
        predicted_body=body,
        predicted_video_path=video_path,
        composite_path=composite_path,
        caption="walk along the route",
        fps=20.0,
    )

    composite_frame = written[str(composite_path)][0]
    assert composite_frame.shape == (160, 256, 3)
    assert len(render_calls) == 2
    target_call, predicted_call = render_calls
    assert target_call[0][2] != video_path
    assert target_call[1]["show_full_trajectory"] is True
    assert target_call[1]["traj_mask"].all()
    assert "show_generated_trajectory" not in target_call[1]
    assert torch.equal(target_call[1]["traj_xz"], target_root[:, [0, 2]])
    assert predicted_call[0][2] == video_path
    assert predicted_call[1]["show_full_trajectory"] is True
    assert predicted_call[1]["show_generated_trajectory"] is True
    assert predicted_call[1]["traj_mask"].all()
    assert torch.equal(predicted_call[1]["traj_xz"], target_root[:, [0, 2]])


@pytest.mark.parametrize(
    ("mode", "expected_final_origin"),
    [("stream", 0), ("rolling", 2)],
)
def test_generation_modes_share_runtime_but_only_rolling_moves_window(
    mode,
    expected_final_origin,
):
    module = _evaluation_module()
    root = torch.zeros(16, 5)
    root[:, 0] = torch.linspace(0.0, 0.3, 16)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    sample = {
        "dataset": "HumanML3D",
        "name": "sample",
        "root_motion": root,
        "body_motion": torch.zeros(16, 259),
        "text_data": [
            {
                "text": "walk",
                "tokens": ["walk/VERB"],
                "start_frame": 0,
                "end_frame": 16,
            }
        ],
    }
    generated = generate_evaluation_sequence(
        module,
        sample,
        mode=mode,
        seed=7,
        frame_count=16,
        dense_xz=True,
        rolling_window_tokens=4,
        max_horizon_token=2,
        num_denoise_steps=2,
        initial_noise_yaw_degrees=0,
    )
    assert generated.hybrid_motion.root_motion.shape == (1, 4, 4, 5)
    assert generated.hybrid_motion.latent_motion.shape == (1, 4, 3)
    assert generated.root_motion.shape == (16, 5)
    assert generated.body_motion.shape == (16, 259)
    assert generated.traces[-1].window_origin_after == expected_final_origin


def test_generation_evaluation_can_override_model_guidance_mode():
    module = _evaluation_module()
    module.model.cfg_mode = "separated"
    module.model._seen_cfg_modes = []
    root = torch.zeros(8, 5)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    sample = {
        "dataset": "HumanML3D",
        "name": "sample",
        "root_motion": root,
        "body_motion": torch.zeros(8, 259),
        "text_data": [
            {
                "text": "walk",
                "tokens": ["walk/VERB"],
                "start_frame": 0,
                "end_frame": 8,
            }
        ],
    }

    generate_evaluation_sequence(
        module,
        sample,
        mode="stream",
        guidance_mode="nocfg",
        seed=7,
        frame_count=8,
        dense_xz=False,
        rolling_window_tokens=4,
        max_horizon_token=2,
        num_denoise_steps=2,
    )

    assert module.model.cfg_mode == "separated"
    assert module.model._seen_cfg_modes
    assert set(module.model._seen_cfg_modes) == {"nocfg"}


def test_generation_evaluation_can_override_joint_cfg_scale():
    module = _evaluation_module()
    module.model.cfg_scale_joint = 9.0
    module.model._seen_cfg_scales = []
    root = torch.zeros(8, 5)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    sample = {
        "dataset": "HumanML3D",
        "name": "sample",
        "root_motion": root,
        "body_motion": torch.zeros(8, 259),
        "text_data": [
            {
                "text": "walk",
                "tokens": ["walk/VERB"],
                "start_frame": 0,
                "end_frame": 8,
            }
        ],
    }

    generate_evaluation_sequence(
        module,
        sample,
        mode="stream",
        guidance_mode="joint",
        cfg_scale_joint=2.0,
        seed=7,
        frame_count=8,
        dense_xz=False,
        rolling_window_tokens=4,
        max_horizon_token=2,
        num_denoise_steps=2,
    )

    assert module.model.cfg_scale_joint == 9.0
    assert module.model._seen_cfg_scales
    assert set(module.model._seen_cfg_scales) == {2.0}
