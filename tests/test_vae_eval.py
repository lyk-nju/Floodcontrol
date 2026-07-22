import torch

from metrics.stream import (
    compute_stream_boundary_metrics,
    compute_stream_vs_offline_metrics,
)
from utils.training.vae.evaluation import (
    MotionSample,
    ReconstructionResult,
    create_rolling_window,
    output_paths,
    reconstruction_metrics,
    rolling_reconstruct,
    stream_reconstruct,
)
from models.vae_wan_1d import BodyVAE
from utils.conditions.vae import BodyPrediction
from tests.vae_helpers import make_vae
from utils.motion_process import recover_joint_positions


def _model() -> BodyVAE:
    return make_vae(
        latent_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=1,
    ).eval()


def _sample(frames: int = 12) -> MotionSample:
    root = torch.zeros(frames, 5)
    root[:, 0] = torch.arange(frames) * 0.01
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    body = torch.randn(frames, 259)
    body[:, 255:] = torch.randint(0, 2, (frames, 4)).float()
    return MotionSample(
        sample_id="sample",
        dataset="HumanML3D",
        root_motion=root,
        body_motion=body,
        body_feature_valid_mask=torch.ones_like(body, dtype=torch.bool),
        previous_root_frame=None,
        fps=20.0,
    )


def test_stream_reconstruction_uses_deterministic_mu_and_matches_offline():
    model = _model()
    sample = _sample()
    first = stream_reconstruct(model, sample, device="cpu")
    second = stream_reconstruct(model, sample, device="cpu")
    assert torch.equal(first.posterior_mu, second.posterior_mu)
    assert torch.equal(
        first.streamed_body.continuous_body,
        second.streamed_body.continuous_body,
    )
    assert first.streamed_body.continuous_body.shape == (1, 12, 255)
    assert first.stream_offline_max_abs <= 1e-5
    metrics = reconstruction_metrics(sample, first)
    assert metrics["frames"] == 12
    assert metrics["tokens"] == 3
    assert metrics["stream_offline_max_abs"] <= 1e-5
    for name in (
        "world_mpjpe_m",
        "source_fk_direct_mpjpe_m",
        "reconstruction_fk_direct_mpjpe_m",
        "reconstruction_fk_target_mpjpe_m",
        "foot_direction_error_deg",
        "foot_reverse_ratio",
        "ankle_toe_length_mae_m",
    ):
        assert torch.isfinite(torch.tensor(metrics[name]))


def test_recover_joint_positions_adds_full_explicit_root_xyz():
    root = torch.tensor([[2.0, 1.2, -3.0, 1.0, 0.0]])
    body = torch.zeros(1, 259)
    body[0, :3] = torch.tensor([0.5, 0.8, -0.25])
    joints = recover_joint_positions(root, body)
    assert joints.shape == (1, 22, 3)
    assert torch.equal(joints[0, 0], torch.tensor([2.0, 1.2, -3.0]))
    assert torch.equal(joints[0, 1], torch.tensor([2.5, 2.0, -3.25]))


def test_stream_metrics_consume_explicit_root_and_body_motion():
    sample = _sample(frames=12)
    boundary = compute_stream_boundary_metrics(
        sample.root_motion,
        sample.body_motion,
        [4, 8, 12],
    )
    assert boundary["n_boundaries"] == 2
    assert abs(boundary["root_jump_mean"] - 0.01) < 1e-7
    parity = compute_stream_vs_offline_metrics(
        sample.root_motion,
        sample.body_motion,
        sample.root_motion,
        sample.body_motion,
    )
    assert parity["feature_l2_mean"] == 0.0
    assert parity["root_ade"] == 0.0


def test_rolling_window_has_fixed_history_and_current_contract():
    latent = torch.arange(23, dtype=torch.float32).reshape(1, 23, 1)
    cold = create_rolling_window(latent, commit_index=0, history_tokens=10)
    assert cold["values"].shape == (1, 11, 1)
    assert not cold["history_mask"].any()
    assert cold["timeline_position_ids"][0, 10] == 0
    assert cold["current_mask"].sum() == 1

    middle = create_rolling_window(latent, commit_index=10, history_tokens=10)
    assert torch.equal(middle["timeline_position_ids"][0], torch.arange(11))
    assert middle["history_mask"][0, :10].all()
    assert middle["current_mask"][0, 10]

    tail = create_rolling_window(latent, commit_index=22, history_tokens=10)
    assert torch.equal(tail["timeline_position_ids"][0, :10], torch.arange(12, 22))
    assert tail["timeline_position_ids"][0, 10] == 22

    warmup = create_rolling_window(latent, commit_index=3, history_tokens=10)
    assert torch.equal(
        warmup["timeline_position_ids"][0, :7], torch.full((7,), -1)
    )
    assert torch.equal(warmup["timeline_position_ids"][0, 7:], torch.arange(4))
    assert warmup["history_mask"].sum() == 3
    assert warmup["current_mask"].sum() == 1


def test_rolling_reconstruction_replays_finite_history_and_checks_cache_parity():
    model = _model()
    sample = _sample(frames=44)
    direct = stream_reconstruct(model, sample, device="cpu")
    rolling = rolling_reconstruct(
        model,
        sample,
        device="cpu",
        history_tokens=1,
        commit_tokens=1,
    )
    assert not torch.allclose(
        rolling.streamed_body.continuous_body,
        direct.streamed_body.continuous_body,
    )
    assert rolling.rolling_reference_max_abs > 0
    assert rolling.stream_offline_max_abs <= 1e-5
    assert torch.equal(rolling.rolling_trace["commit_token"], torch.arange(11))
    assert rolling.rolling_trace["timeline_position_ids"].shape == (11, 2)
    assert rolling.rolling_trace["history_mask"][0].sum() == 0
    assert rolling.rolling_trace["history_mask"][1:, 0].all()
    assert not rolling.rolling_trace["history_mask"][:, 1].any()
    assert rolling.rolling_trace["current_mask"][:, 1].all()
    assert not rolling.rolling_trace["current_mask"][:, 0].any()
    metrics = reconstruction_metrics(sample, rolling)
    assert metrics["history_tokens"] == 1
    assert metrics["rolling_steps"] == 11
    assert metrics["rolling_stream_max_abs"] > 0
    assert metrics["cache_window_offline_max_abs"] <= 1e-5


def test_rolling_full_decoder_context_matches_persistent_stream():
    model = _model()
    sample = _sample(frames=44)
    result = rolling_reconstruct(
        model,
        sample,
        device="cpu",
        history_tokens=model.decoder_context_tokens,
        commit_tokens=1,
    )
    assert result.rolling_reference_max_abs <= 1e-5


def test_reconstruction_skating_metrics_use_position_transitions_and_masks():
    sample = _sample(frames=12)
    sample.body_motion.zero_()
    sample.body_motion[:, 255] = 1.0
    continuous = torch.zeros(1, 12, 255)
    continuous[0, :, :63].reshape(12, 21, 3)[:, 6, 0] = (
        torch.arange(12) * 0.1
    )
    logits = torch.full((1, 12, 4), -100.0)
    logits[..., 0] = 100.0
    body = BodyPrediction(continuous, logits)
    result = ReconstructionResult(
        protocol="test",
        posterior_mu=torch.zeros(1, 3, 8),
        local_root_motion=torch.zeros(1, 3, 4, 4),
        local_root_valid_mask=torch.ones(1, 3, 4, 4, dtype=torch.bool),
        streamed_body=body,
        offline_body=body,
        stream_offline_max_abs=0.0,
    )
    metrics = reconstruction_metrics(sample, result)
    assert metrics["gt_contact_position_skating_mps"] > 0
    assert metrics["predicted_contact_position_skating_mps"] > 0
    assert metrics["gt_contact_velocity_feature_mps"] == 0


def test_output_layout_separates_original_and_reconstruction(tmp_path):
    paths = output_paths(tmp_path, "humanml3d", "vae_body259_run", "sample")
    model_root = tmp_path / "humanml3d/vae_body259_run"
    assert paths["original_video"] == model_root / "video/original/sample.mp4"
    assert paths["reconstruction_video"] == model_root / "video/reconstruction/sample.mp4"
    assert paths["original_motion"] == model_root / "motion/original/sample.npz"
    assert paths["reconstruction_motion"] == model_root / "motion/reconstruction/sample.npz"
