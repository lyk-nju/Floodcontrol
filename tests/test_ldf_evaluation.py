from __future__ import annotations

import json
import os
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
from utils.training.ldf.evaluation.artifacts import save_dense_xz_sample
from utils.training.ldf.evaluation.generation import (
    compile_evaluation_prompt,
    generate_evaluation_sequence,
)
from utils.training.ldf.evaluation.runner import (
    LDFEvaluationRunner,
    _all_gather_objects,
    _distributed_barrier,
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
    del kwargs
    root_velocity = torch.zeros_like(inputs.noisy_motion.root_motion)
    latent_velocity = torch.zeros_like(inputs.noisy_motion.latent_motion)
    local = torch.zeros(*root_velocity.shape[:3], 4, device=root_velocity.device)
    valid = torch.ones_like(local, dtype=torch.bool)
    return LDFPrediction(
        HybridMotion(root_velocity, latent_velocity),
        inputs.noisy_motion.root_motion,
        local,
        valid,
    )


def _evaluation_module():
    model = LDF(
        latent_dim=3,
        root_mean=[0] * 5,
        root_std=[1] * 5,
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
    callback.runner = types.SimpleNamespace(
        maybe_run=calls.append,
        validate_text_coverage=coverage_calls.append,
    )
    module = object()

    callback.on_fit_start(types.SimpleNamespace(is_global_zero=True), module)
    callback.on_fit_start(types.SimpleNamespace(is_global_zero=False), module)
    assert coverage_calls == [module, module]

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


def test_dense_xz_metrics_report_time_aligned_and_path_errors():
    target = torch.zeros(8, 5)
    target[:, 0] = torch.arange(8) * 0.1
    predicted = target.clone()
    predicted[:, 2] += 0.1
    metrics = compute_dense_xz_metrics(
        predicted,
        target,
        segment_frames=4,
    )
    assert metrics["ade"] == pytest.approx(0.1)
    assert metrics["fde"] == pytest.approx(0.1)
    assert metrics["mse"] == pytest.approx(0.01)
    assert metrics["traj_fail_20cm"] == 0.0
    assert metrics["path_arc_ade"] == pytest.approx(0.1)
    assert len(metrics["segment_mse"]) == 2


def test_dense_xz_summary_keeps_modes_and_segment_slots_separate():
    summary = summarize_dense_xz_records(
        [
            {"ade": 0.1, "fde": 0.2, "segment_mse": [0.01, 0.02]},
            {"ade": 0.3, "fde": 0.4, "segment_mse": [0.03, None]},
        ]
    )
    assert summary["num_samples"] == 2
    assert summary["ade_mean"] == pytest.approx(0.2)
    assert summary["fde_std"] == pytest.approx(0.1)
    assert summary["segment_mse_per_slot"] == pytest.approx([0.02, 0.02])


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


def test_dense_xz_artifacts_follow_floodnet_layout(tmp_path):
    root = torch.zeros(8, 5)
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    body = torch.zeros(8, 265)
    dirs = save_dense_xz_sample(
        save_dir=tmp_path,
        dataset="HumanML3D",
        probe="dense_xz_stream",
        step_tag="step_010000",
        sample_id="sample",
        caption="walk",
        normalized_root=torch.zeros(2, 4, 5),
        normalized_latent=torch.zeros(2, 128),
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


def test_dense_xz_video_contains_generated_motion_and_trajectory_panel(
    tmp_path,
    monkeypatch,
):
    frames = [
        np.full((128, 128, 3), fill_value=value, dtype=np.uint8)
        for value in (32, 64)
    ]
    written: dict[str, list[np.ndarray]] = {}

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

    monkeypatch.setattr(
        evaluation_artifacts,
        "render_motion_video",
        lambda *args, **kwargs: None,
    )
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
    body = torch.zeros(2, 265)
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

    video_frame = written[str(video_path)][0]
    composite_frame = written[str(composite_path)][0]
    assert video_frame.shape == (160, 256, 3)
    assert composite_frame.shape == (160, 384, 3)
    trajectory_panel = video_frame[32:, 128:]
    assert np.any(np.all(trajectory_panel == (20, 150, 20), axis=-1))
    assert np.any(np.all(trajectory_panel == (210, 40, 40), axis=-1))


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
        "body_motion": torch.zeros(16, 265),
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
        rebase_on_roll=False,
    )
    assert generated.normalized_motion.root_motion.shape == (1, 4, 4, 5)
    assert generated.normalized_motion.latent_motion.shape == (1, 4, 3)
    assert generated.root_motion.shape == (16, 5)
    assert generated.body_motion.shape == (16, 265)
    assert generated.traces[-1].window_origin_after == expected_final_origin
