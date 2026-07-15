from pathlib import Path
import inspect

import pytest

from models.diffusion_forcing_wan import LDF
from models.vae_wan_1d import BodyVAE
from tests.vae_helpers import make_vae
from train_ldf import _validate_training_config, main as train_main
from utils.initialize import instantiate_target, load_config
from web_demo.runtime.model_loader import WEB_RUNTIME_BLOCKER, load_model_bundle


ROOT = Path(__file__).resolve().parents[1]


def test_formal_ldf_config_uses_the_vae_as_contract_source():
    cfg = load_config(str(ROOT / "configs" / "ldf.yaml"))
    assert cfg.status == "training_ready"
    assert cfg.model.target == "models.diffusion_forcing_wan.LDF"
    assert cfg.vae.target == "models.vae_wan_1d.BodyVAE"
    assert cfg.vae.params.latent_dim == 128
    assert "encoder_context_tokens" not in cfg.data
    assert cfg.data.min_frames == 40
    assert cfg.data.max_frames == 200
    assert cfg.data.cold_start_probability == 0.1
    assert cfg.data.length_bucket_frames == 20
    assert cfg.text_embeddings_path.endswith("HumanML3D_motion/t5_text_embeddings.pt")
    assert cfg.validation.continuation_span_frames == 40
    assert cfg.validation.continuation_history_tokens == 5
    assert cfg.validation.self_forcing_steps == 5
    assert cfg.self_forcing.phase_start_step == 300000
    assert cfg.self_forcing.phase_steps == 200000
    assert list(cfg.self_forcing.k_schedule) == [[0.0, 2], [0.4, 3], [0.7, 5]]
    assert cfg.self_forcing.teacher_replay[2] == 0.2
    for injected_name in (
        "latent_dim",
        "root_mean",
        "root_std",
        "local_root_mean",
        "local_root_std",
    ):
        assert injected_name not in cfg.model.params


def test_mixed_ldf_config_uses_the_same_prompt_and_model_contract():
    cfg = load_config(str(ROOT / "configs" / "ldf_multi.yaml"))
    human_cfg = load_config(str(ROOT / "configs" / "ldf.yaml"))
    assert cfg.status == "training_ready"
    assert cfg.data.target == "datasets.multi.MultiDataset"
    assert [item.target for item in cfg.data.datasets] == [
        "datasets.humanml3d.HumanML3DDataset",
        "datasets.babel.BABELDataset",
    ]
    assert all(item.text_path == "texts" for item in cfg.data.datasets)
    assert cfg.model.params.text_len == cfg.text_encoder.text_len == 128
    assert cfg.self_forcing.enabled is False
    assert cfg.text_embeddings_path.endswith(
        "HumanML3D_BABEL_t5_text_embeddings.pt"
    )
    assert cfg.root_stats_path == human_cfg.root_stats_path
    assert cfg.validation.continuation_span_frames == 40


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
    assert "BLOCKED_ON_LDF_TRAINING" not in source
    cfg = load_config(str(ROOT / "configs" / "ldf.yaml"))
    with pytest.raises(RuntimeError, match="LDF_ROOT_STATISTICS_REQUIRED"):
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
