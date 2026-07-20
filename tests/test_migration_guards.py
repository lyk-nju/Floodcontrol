from pathlib import Path
import inspect
import subprocess
import sys

import numpy as np
import pytest

from models.diffusion_forcing_wan import LDF
from models.vae_wan_1d import BodyVAE
from tests.vae_helpers import make_vae
from train_ldf import _validate_training_config, main as train_main
from utils.initialize import instantiate_target, load_config
from utils.training.ldf.window import resolve_self_forcing_k
from web_demo.runtime.model_loader import WEB_RUNTIME_BLOCKER, load_model_bundle


ROOT = Path(__file__).resolve().parents[1]
REMOTE_LDF_CONFIG = ROOT / "configs" / "ldf.yaml"
LOCAL_LDF_CONFIG = ROOT / "configs" / "ldf_s5.yaml"


def test_low_level_ldf_import_does_not_load_runtime_or_lightning():
    code = """
import sys
import utils.training.ldf.flow

for forbidden in (
    'utils.training.ldf.lightning_module',
    'utils.training.ldf.evaluation',
    'utils.training.ldf.evaluation.runner',
    'utils.inference',
):
    assert forbidden not in sys.modules, forbidden
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
    )


def test_formal_ldf_config_uses_the_vae_as_contract_source():
    cfg = load_config(str(REMOTE_LDF_CONFIG))
    assert "status" not in cfg.config
    assert cfg.wandb_info.project == "Floodcontrol"
    assert cfg.model.target == "models.diffusion_forcing_wan.LDF"
    assert cfg.vae.target == "models.vae_wan_1d.BodyVAE"
    assert cfg.vae.params.latent_dim == 128
    assert "fps" not in cfg.vae.params
    assert "encoder_context_tokens" not in cfg.data
    assert cfg.data.min_frames == 20
    assert cfg.data.max_frames == 200
    assert "cold_start_probability" not in cfg.data
    assert "length_bucket_frames" not in cfg.data
    assert cfg.training.text_dropout == pytest.approx(0.1)
    assert cfg.training.constraint_dropout == pytest.approx(0.1)
    assert cfg.training.window.max_tokens == 50
    assert cfg.training.window.generation_tokens == 5
    assert cfg.training.window.sampling == "random_generation_start"
    assert cfg.training.max_horizon_token == 45
    assert cfg.training.constraint_sampling.dense_probability == pytest.approx(1.0)
    assert cfg.training.constraint_sampling.waypoint_probability == pytest.approx(0.0)
    assert cfg.training.constraint_sampling.goal_probability == pytest.approx(0.0)
    assert cfg.training.constraint_sampling.max_waypoint_count == 4
    assert cfg.data.text_embeddings_path.endswith(
        "HumanML3D_motion/t5_text_embeddings.pt"
    )
    assert list(cfg.data.text_meta_paths) == [
        str(Path(cfg.data.train_meta_paths[0]).with_name("all.txt"))
    ]
    assert "continuation_span_frames" not in cfg.validation
    assert "loss_probes" not in cfg.validation
    assert cfg.validation.generation.enabled is True
    assert cfg.validation.generation.modes
    assert set(cfg.validation.generation.modes) <= {"stream", "rolling"}
    assert cfg.validation.generation.max_horizon_token == 10
    assert cfg.validation.generation.rolling.window_tokens == 50
    assert cfg.validation.dense_xz.enabled is True
    assert cfg.validation.dense_xz.probe == "dense_xz"
    assert cfg.data.test_probe_meta_paths.dense_xz[0].endswith(
        "HumanML3D_motion/test_min.txt"
    )
    assert cfg.validation.t2m.enabled is True
    assert "max_samples" not in cfg.validation.t2m
    assert "enabled" not in cfg.self_forcing
    assert "phase_start_step" not in cfg.self_forcing
    assert "phase_steps" not in cfg.self_forcing
    assert cfg.self_forcing.cold_start_replay == pytest.approx(0.1)
    assert list(cfg.self_forcing.k_schedule) == [
        [0, 1],
        [100000, 2],
        [200000, 3],
        [300000, 5],
    ]
    assert dict(cfg.self_forcing.teacher_replay) == {2: 0.1, 3: 0.1, 5: 0.1}
    assert cfg.trainer.max_steps == 500000
    assert cfg.trainer.devices == 8
    assert cfg.data.train_batch_size == 64
    assert cfg.data.num_workers == 2
    assert cfg.dirs.raw_data.startswith("/data/home/shengqiuProf_user_yuankai/")
    assert cfg.resume_ckpt is None
    assert cfg.test_ckpt is None
    assert cfg.loss.rollout_weight == pytest.approx(1.0)
    assert cfg.loss.offpath_beta_min == pytest.approx(0.1)
    assert cfg.loss.root_boundary_weight == pytest.approx(0.0)
    assert cfg.loss.root_heading_weight == pytest.approx(0.1)
    assert cfg.loss.body_weight == pytest.approx(3.0)
    assert cfg.model.params.cfg_mode == "joint"
    assert cfg.root_statistics.window_tokens == 50
    assert cfg.root_statistics.generation_tokens == 5
    assert cfg.root_statistics.anchor_sampling == "uniform_legal_history"
    assert (
        cfg.lr_scheduler.target
        == "diffusers.optimization.get_cosine_schedule_with_warmup"
    )
    assert cfg.lr_scheduler.params.num_warmup_steps == 5_000
    assert cfg.lr_scheduler.params.num_training_steps == cfg.trainer.max_steps
    for injected_name in (
        "latent_dim",
        "root_mean",
        "root_std",
        "local_root_mean",
        "local_root_std",
    ):
        assert injected_name not in cfg.model.params


def test_formal_t2m_evaluation_uses_nocfg():
    cfg = load_config(str(REMOTE_LDF_CONFIG))
    assert cfg.validation.t2m.enabled is True
    assert cfg.validation.t2m.cfg_mode == "nocfg"


def test_mixed_ldf_config_uses_the_same_prompt_and_model_contract():
    cfg = load_config(str(ROOT / "configs" / "ldf_multi.yaml"))
    human_cfg = load_config(str(LOCAL_LDF_CONFIG))
    _validate_training_config(cfg)
    assert "status" not in cfg.config
    assert cfg.wandb_info.project == "Floodcontrol"
    assert cfg.data.target == "datasets.multi.MultiDataset"
    assert [item.target for item in cfg.data.datasets] == [
        "datasets.humanml3d.HumanML3DDataset",
        "datasets.babel.BABELDataset",
    ]
    assert all(item.text_path == "texts" for item in cfg.data.datasets)
    assert all(
        len(item.text_meta_paths) == 1
        and Path(item.text_meta_paths[0]).name == "all.txt"
        for item in cfg.data.datasets
    )
    assert cfg.model.params.text_len == cfg.text_encoder.text_len == 128
    assert cfg.self_forcing == human_cfg.self_forcing
    assert cfg.data.text_embeddings_path.endswith(
        "HumanML3D_BABEL_t5_text_embeddings.pt"
    )
    assert cfg.data.root_stats_path == human_cfg.data.root_stats_path
    assert "continuation_span_frames" not in cfg.validation
    assert cfg.training.constraint_dropout == pytest.approx(0.1)
    assert cfg.training.window == human_cfg.training.window
    assert cfg.training.max_horizon_token == human_cfg.training.max_horizon_token
    assert cfg.training.constraint_sampling == human_cfg.training.constraint_sampling
    assert cfg.model.params == human_cfg.model.params
    assert cfg.loss == human_cfg.loss
    assert cfg.optimizer == human_cfg.optimizer
    assert cfg.lr_scheduler == human_cfg.lr_scheduler


def test_remote_self_forcing_recipe_uses_resume_stable_absolute_boundaries():
    cfg = load_config(str(REMOTE_LDF_CONFIG))
    schedule = cfg.self_forcing.k_schedule

    assert resolve_self_forcing_k(0, schedule) == 1
    assert resolve_self_forcing_k(99_999, schedule) == 1
    assert resolve_self_forcing_k(100_000, schedule) == 2
    assert resolve_self_forcing_k(199_999, schedule) == 2
    assert resolve_self_forcing_k(200_000, schedule) == 3
    assert resolve_self_forcing_k(299_999, schedule) == 3
    assert resolve_self_forcing_k(300_000, schedule) == 5
    assert resolve_self_forcing_k(499_999, schedule) == 5


def test_tiny_vae_config_instantiates_public_body_vae():
    model = make_vae(
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        with_latent_stats=False,
    )
    assert isinstance(model, BodyVAE)


def test_body_vae_lifecycle_boundaries_are_explicit():
    parameters = inspect.signature(BodyVAE.__init__).parameters
    assert "motion_stats_path" in parameters
    assert "latent_stats_path" in parameters
    assert "allow_identity_statistics" not in parameters
    assert "require_latent_statistics" not in parameters
    assert not hasattr(BodyVAE, "bind_tokenizer_identity")
    assert not hasattr(BodyVAE, "encode_window")
    assert not hasattr(BodyVAE, "stream_decode_step")
    assert not hasattr(BodyVAE, "snapshot_decoder_state")
    assert "normalized_latent" not in inspect.signature(
        BodyVAE.detokenize_step
    ).parameters

    cfg = load_config(str(ROOT / "configs" / "vae.yaml"))
    assert cfg.model.target == "models.vae_wan_1d.BodyVAE"
    assert not (ROOT / "utils" / "vae_tokenizer.py").exists()
    assert not (ROOT / "utils" / "inference" / "vae_decoder.py").exists()


def test_legacy_vae_config_and_class_are_removed():
    assert not (ROOT / "configs" / "vae_wan_1d.yaml").exists()
    import models.vae_wan_1d as module

    assert not hasattr(module, "VAEWanModel")


def test_removed_motion_process_apis_do_not_return():
    import utils.motion_process as module

    for name in (
        "recover_root_rot_pos",
        "extract_root_xz_phi_features_263",
        "extract_root_traj_feats_7d_263",
        "replace_root_channels_263_from_7d",
        "replace_root_channels_263_window_from_7d",
        "StreamJointRecovery263",
        "convert_motion_to_joints",
        "build_physical_7d_from_5d",
    ):
        assert not hasattr(module, name)


def test_motion_process_owns_physical_contract_and_token_frame_owns_time_contract():
    import utils.conditions.ldf as ldf_contract
    import utils.conditions.vae as vae_contract
    import utils.motion_process as motion_process
    import utils.token_frame as token_frame

    assert vae_contract.ROOT_DIM == motion_process.ROOT_DIM
    assert ldf_contract.LOCAL_ROOT_DIM == motion_process.LOCAL_ROOT_DIM
    assert motion_process.FRAMES_PER_TOKEN == token_frame.FRAMES_PER_TOKEN
    assert vae_contract.FRAMES_PER_TOKEN == token_frame.FRAMES_PER_TOKEN
    assert ldf_contract.FRAMES_PER_TOKEN == token_frame.FRAMES_PER_TOKEN
    assert "FRAMES_PER_TOKEN =" not in (ROOT / "utils" / "motion_process.py").read_text()
    assert not hasattr(ldf_contract, "derive_local_root_motion")
    assert not hasattr(ldf_contract, "project_root_heading")
    assert callable(motion_process.recover_local_root)
    assert callable(motion_process.project_root_heading)


def test_motion_processing_has_one_canonical_runtime_module():
    assert not (ROOT / "utils" / "motion_representation.py").exists()
    assert not (ROOT / "utils" / "traj_batch.py").exists()
    assert not (ROOT / "utils" / "path_arclength.py").exists()
    for relative_path in (
        "utils/inference/stream_runtime",
        "utils/inference/runtime_update",
        "metrics/traj.py",
        "datasets/generate.py",
    ):
        path = ROOT / relative_path
        assert not path.exists() or not any(
            child.suffix == ".py" for child in path.rglob("*.py")
        )


def test_ldf_training_has_one_canonical_solver_and_condition_stack():
    for relative_path in (
        "utils/training/ldf/batch.py",
        "utils/training/ldf/self_forcing.py",
        "utils/training/ldf/rollout.py",
        "tools/compute_z_stats.py",
    ):
        assert not (ROOT / relative_path).exists()

    import utils.conditions.ldf as ldf_contract

    assert not hasattr(ldf_contract, "create_window_condition")


def test_offline_artifact_builder_has_distinct_name():
    assert not (ROOT / "utils" / "motion_artifact.py").exists()
    assert (ROOT / "tools" / "build_motion_artifact.py").is_file()
    assert not (ROOT / "tools" / "motion_artifact.py").exists()


def test_coordinate_transform_is_the_only_runtime_coordinate_module():
    import utils.coordinate_transform as coordinate_transform

    assert not (ROOT / "utils" / "local_frame.py").exists()
    for name in (
        "heading_to_direction",
        "rotate_vectors_world_to_local",
        "rotate_vectors_local_to_world",
        "transform_points_world_to_local",
        "transform_points_local_to_world",
    ):
        assert callable(getattr(coordinate_transform, name))
    for name in (
        "canonicalize_5d",
        "canonicalize_7d",
        "uncanonicalize_7d",
        "root_quat_to_physical_yaw",
    ):
        assert not hasattr(coordinate_transform, name)


def test_training_entry_uses_the_public_ldf_training_stack():
    assert callable(train_main)
    source = inspect.getsource(train_main)
    assert "LDFLightningModule" in source
    assert "create_dataloaders" in source
    assert 'trainer_config["use_distributed_sampler"] = False' in source
    assert "BLOCKED_ON_LDF_TRAINING" not in source
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    _validate_training_config(cfg)
    assert cfg.training.constraint_dropout == pytest.approx(0.1)
    assert cfg.training.window.max_tokens == 50
    assert cfg.training.window.generation_tokens == 5
    assert cfg.training.max_horizon_token == 45

    # Scheduled dense-XZ/video/T2M evaluation is sharded by the training
    # callback and therefore remains legal for a multi-device DDP run.
    cfg.config.trainer.devices = 8
    cfg.config.trainer.strategy = "ddp"
    _validate_training_config(cfg)


def test_training_entry_rejects_missing_xz_lookahead():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.training.max_horizon_token = 0
    with pytest.raises(RuntimeError, match="LDF_XZ_CONSTRAINT_REQUIRED"):
        _validate_training_config(cfg)


def test_training_entry_accepts_root_statistics_without_sampling_metadata(tmp_path):
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    root_stats = tmp_path / "root_stats.npz"
    np.savez(root_stats, root_mean=np.zeros(5), root_std=np.ones(5))
    cfg.config.data.root_stats_path = str(root_stats)
    cfg.config.root_statistics = None

    _validate_training_config(cfg)


def test_training_entry_rejects_invalid_constraint_dropout():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.training.constraint_dropout = 1.1
    with pytest.raises(ValueError, match="constraint_dropout"):
        _validate_training_config(cfg)


def test_training_entry_rejects_invalid_constraint_sampling():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.training.constraint_sampling.goal_probability = 0.5
    with pytest.raises(ValueError, match="must sum to one"):
        _validate_training_config(cfg)

    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.training.constraint_sampling.max_waypoint_count = 0
    with pytest.raises(ValueError, match="max_waypoint_count"):
        _validate_training_config(cfg)


def test_training_entry_rejects_self_forcing_that_exceeds_window_budget():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.self_forcing.k_schedule = [[0, 1], [1, 47]]
    cfg.config.self_forcing.teacher_replay = {47: 0.1}
    with pytest.raises(ValueError, match="self-forcing rollout"):
        _validate_training_config(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("phase_start_step", 0, "has been removed"),
        ("phase_steps", 1, "has been removed"),
        ("k_schedule", [[0, 1], [0, 2]], "strictly increasing"),
        ("k_schedule", [[0, 1], [1.2, 2]], "must be an integer"),
        ("k_schedule", [[0, 1], [2, 0]], "at least 1"),
        ("teacher_replay", {2: 2.0, 5: 0.1}, "probabilities"),
        ("teacher_replay", {2: 0.2, 4: 0.1}, "exactly match"),
        ("cold_start_replay", 1.1, "cold_start_replay"),
    ],
)
def test_training_entry_rejects_invalid_self_forcing_contract(
    field,
    value,
    message,
):
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.self_forcing[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_training_config(cfg)


def test_training_entry_rejects_schedule_stage_outside_training_run():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.self_forcing.k_schedule.append([cfg.trainer.max_steps, 6])
    cfg.config.self_forcing.teacher_replay[6] = 0.0
    with pytest.raises(ValueError, match="start steps must be smaller"):
        _validate_training_config(cfg)


def test_training_entry_rejects_removed_self_forcing_enabled_switch():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    cfg.config.self_forcing.enabled = False
    with pytest.raises(ValueError, match="enabled has been removed"):
        _validate_training_config(cfg)


def test_training_entry_requires_unified_self_forcing_curriculum():
    cfg = load_config(str(LOCAL_LDF_CONFIG))
    del cfg.config.self_forcing
    with pytest.raises(ValueError, match="curriculum configuration is required"):
        _validate_training_config(cfg)


def test_web_entry_is_explicitly_blocked():
    with pytest.raises(RuntimeError, match="BLOCKED_ON_LDF_CHECKPOINT"):
        load_model_bundle(None)
    assert "InferenceSession" in WEB_RUNTIME_BLOCKER
    assert "hybrid LDF" in WEB_RUNTIME_BLOCKER


def test_web_runtime_removed_duplicate_legacy_state_owners():
    for relative_path in (
        "web_demo/model_manager.py",
        "web_demo/bootstrap.py",
        "web_demo/runtime/frame_buffer.py",
        "web_demo/runtime/generation_worker.py",
        "web_demo/runtime/state.py",
        "web_demo/runtime/trajectory_controller.py",
        "web_demo/services/session_service.py",
    ):
        assert not (ROOT / relative_path).exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "models/diffusion_forcing_wan_tiny.py",
        "models/tools/wan_controlnet.py",
        "models/tools/traj_encoder.py",
        "models/root_" + "refiner.py",
        "utils/conditions/root_" + "refiner.py",
        "utils/inference/root_plan.py",
        "utils/inference/stream_generator.py",
    ],
)
def test_removed_architecture_files_are_physically_absent(relative_path):
    assert not (ROOT / relative_path).exists()


def test_inference_uses_hybrid_session_without_legacy_root_recovery():
    import utils.inference as inference

    for relative_path in (
        "utils/inference/condition_manager.py",
        "utils/inference/route_condition.py",
        "utils/inference/text_condition.py",
        "utils/inference/timeline.py",
    ):
        assert not (ROOT / relative_path).exists()
    for public_name in (
        "InferenceSession",
        "InferenceConditionCompiler",
        "RoutePlan",
        "TextTimeline",
        "RootObservation",
    ):
        assert hasattr(inference, public_name)
    for legacy_name in (
        "ConditionManager",
        "RootTimeline",
        "RouteConditionState",
        "TextConditionState",
    ):
        assert not hasattr(inference, legacy_name)


def test_initialize_uses_explicit_public_names_only():
    import utils.initialize as initialize

    for old_name in (
        "Config",
        "instantiate",
        "get_function",
        "save_config_and_codes",
        "print_model_size",
        "check_state_dict",
        "get_shared_run_time",
    ):
        assert not hasattr(initialize, old_name)
    for public_name in (
        "ProjectConfig",
        "instantiate_target",
        "resolve_function",
        "save_run_snapshot",
        "log_model_parameters",
        "log_state_dict_summary",
        "get_shared_run_timestamp",
    ):
        assert hasattr(initialize, public_name)


def test_legacy_training_step_semantics_are_removed():
    import utils.training as training

    assert not (ROOT / "utils" / "training" / "step_semantics.py").exists()
    assert not (ROOT / "utils" / "training" / "module_step.py").exists()
    for old_name in (
        "StepSemantics",
        "CheckpointStepInfo",
        "build_step_semantics",
        "compute_step_semantics",
        "ckpt_step_info",
        "load_resume_step_offset",
        "resolve_runtime_max_steps",
        "resolve_scheduler_steps",
    ):
        assert not hasattr(training, old_name)
