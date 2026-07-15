import torch

from eval.vae.evaluate_reconstruction import (
    MotionSample,
    _output_paths,
    body_to_global_joints,
    create_rolling_window,
    reconstruction_metrics,
    rolling_reconstruct,
    stream_reconstruct,
)
from models.vae_wan_1d import BodyVAE


def _model() -> BodyVAE:
    return BodyVAE(
        latent_dim=8,
        hidden_dim=16,
        encoder_layers=1,
        decoder_layers=1,
        allow_identity_statistics=True,
        require_latent_statistics=False,
    ).eval()


def _sample(frames: int = 12) -> MotionSample:
    root = torch.zeros(frames, 5)
    root[:, 0] = torch.arange(frames) * 0.01
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    body = torch.randn(frames, 265)
    body[:, 261:] = torch.randint(0, 2, (frames, 4)).float()
    return MotionSample(
        sample_id="sample",
        dataset="HumanML3D_motion",
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
    assert first.streamed_body.continuous_body.shape == (1, 12, 261)
    assert first.stream_offline_max_abs <= 1e-5
    metrics = reconstruction_metrics(sample, first)
    assert metrics["frames"] == 12
    assert metrics["tokens"] == 3
    assert metrics["stream_offline_max_abs"] <= 1e-5


def test_body_to_global_joints_uses_explicit_root_xz_and_global_height():
    root = torch.tensor([[2.0, 1.2, -3.0, 1.0, 0.0]])
    body = torch.zeros(1, 265)
    body[0, :3] = torch.tensor([0.5, 0.8, -0.25])
    joints = body_to_global_joints(root, body)
    assert joints.shape == (1, 22, 3)
    assert torch.equal(joints[0, 0], torch.tensor([2.0, 1.2, -3.0]))
    assert torch.equal(joints[0, 1], torch.tensor([2.5, 0.8, -3.25]))


def test_rolling_window_has_fixed_10_history_10_future_contract():
    latent = torch.arange(23, dtype=torch.float32).reshape(1, 23, 1)
    cold = create_rolling_window(
        latent, commit_index=0, history_tokens=10, future_tokens=10
    )
    assert cold["values"].shape == (1, 20, 1)
    assert not cold["history_mask"].any()
    assert torch.equal(
        cold["timeline_position_ids"][0, 10:], torch.arange(10)
    )

    middle = create_rolling_window(
        latent, commit_index=10, history_tokens=10, future_tokens=10
    )
    assert torch.equal(
        middle["timeline_position_ids"][0], torch.arange(20)
    )
    assert middle["history_mask"][0, :10].all()
    assert middle["future_mask"][0, 10:].all()

    tail = create_rolling_window(
        latent, commit_index=22, history_tokens=10, future_tokens=10
    )
    assert torch.equal(
        tail["timeline_position_ids"][0, :10], torch.arange(12, 22)
    )
    assert tail["timeline_position_ids"][0, 10] == 22
    assert torch.equal(tail["timeline_position_ids"][0, 11:], torch.full((9,), -1))
    assert tail["future_mask"].sum() == 1


def test_rolling_reconstruction_commits_each_token_once_and_matches_stream():
    model = _model()
    sample = _sample(frames=44)
    direct = stream_reconstruct(model, sample, device="cpu")
    rolling = rolling_reconstruct(
        model,
        sample,
        device="cpu",
        history_tokens=10,
        future_tokens=10,
        commit_tokens=1,
    )
    assert torch.equal(
        rolling.streamed_body.continuous_body,
        direct.streamed_body.continuous_body,
    )
    assert torch.equal(rolling.rolling_trace["commit_token"], torch.arange(11))
    assert rolling.rolling_trace["timeline_position_ids"].shape == (11, 20)
    metrics = reconstruction_metrics(sample, rolling)
    assert metrics["history_tokens"] == 10
    assert metrics["future_tokens"] == 10
    assert metrics["rolling_steps"] == 11


def test_output_layout_separates_original_and_reconstruction(tmp_path):
    paths = _output_paths(tmp_path, "humanml3d", "sample")
    assert paths["original_video"] == tmp_path / "humanml3d/video/original/sample.mp4"
    assert paths["reconstruction_video"] == tmp_path / "humanml3d/video/reconstruction/sample.mp4"
    assert paths["original_motion"] == tmp_path / "humanml3d/motion/original/sample.npz"
    assert paths["reconstruction_motion"] == tmp_path / "humanml3d/motion/reconstruction/sample.npz"
