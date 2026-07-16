from __future__ import annotations

import types
from dataclasses import replace

import numpy as np
import pytest
import torch

from models.diffusion_forcing_wan import LDF
from tests.vae_helpers import make_vae
from utils.conditions.ldf import HybridMotion, LDFPrediction
from utils.inference.condition import (
    InferenceConditionCompiler,
    RootObservation,
    RootObservationTimeline,
)
from utils.inference.geometry import (
    assign_times_by_arclength,
    sample_timed_route,
    validate_route_times,
)
from utils.inference.route import RouteEndBehavior, RoutePlan, RouteReference
from utils.inference.session import GuidanceConfig, InferenceConfig, InferenceSession
from utils.inference.text import TextEmbeddingCache, TextTimeline


def make_ldf(*, latent_dim=3):
    return LDF(
        latent_dim=latent_dim,
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
        noise_steps=4,
    ).eval()


def encode_texts(texts, device):
    return [
        torch.full((1, 4), float(sum(map(ord, text)) % 97), device=device)
        for text in texts
    ]


def zero_prediction(self, inputs, **kwargs):
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


def make_initial_noise(*, root_x=0.0, tokens=6, latent_dim=3):
    root = torch.zeros(1, tokens, 4, 5)
    root[..., 0] = float(root_x)
    root[..., 3] = 1.0
    latent = torch.zeros(1, tokens, latent_dim)
    return HybridMotion(root, latent)


def make_session(*, root_x=0.0, guidance=None, rolling=True):
    ldf = make_ldf()
    ldf.predict_with_cfg = types.MethodType(zero_prediction, ldf)
    vae = make_vae(
        latent_dim=3,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
    ).eval()
    session = InferenceSession(
        ldf=ldf,
        body_vae=vae,
        text_encoder=encode_texts,
        config=InferenceConfig(
            window_tokens=6,
            max_horizon_token=2,
            rolling=rolling,
        ),
        guidance=guidance,
        initial_world_xz=(10.0, 20.0),
        initial_noise=make_initial_noise(root_x=root_x),
    )
    return session


def test_geometry_requires_explicit_time_units_and_end_behavior():
    points = np.asarray([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    times = assign_times_by_arclength(points, 2.0)
    sampled, valid = sample_timed_route(
        times,
        points,
        np.asarray([-0.1, 1.0, 3.0], dtype=np.float32),
        hold_after_end=False,
    )
    assert valid.tolist() == [False, True, False]
    assert np.allclose(sampled[1], [1.0, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_route_times(np.asarray([0.0, 0.0]), point_count=2)


def test_relative_route_is_resolved_to_world_once():
    route = RoutePlan(
        times=np.asarray([0.0, 1.0]),
        points_xz=np.asarray([[0.0, 0.0], [1.0, 2.0]]),
        start_token=3,
    ).resolve_world(RouteReference.RELATIVE_TO_ACTOR, np.asarray([10.0, 20.0]))
    assert np.allclose(route.points_xz, [[10.0, 20.0], [11.0, 22.0]])
    sampled, valid = route.sample_frames(np.asarray([12, 16]), fps=20.0)
    assert valid.all()
    assert np.allclose(sampled[0], [10.0, 20.0])


def test_text_timeline_supports_token_aligned_updates_and_cache_reuse():
    timeline = TextTimeline("walk")
    timeline.update("turn", start_token=3)
    assert timeline.resolve([0, 2, 3, 8]) == ["walk", "walk", "turn", "turn"]
    calls = []

    def encoder(texts, device):
        calls.append(tuple(texts))
        return encode_texts(texts, device)

    cache = TextEmbeddingCache(encoder)
    cache.encode(["walk", "turn", "walk"], device=torch.device("cpu"))
    cache.encode(["turn"], device=torch.device("cpu"))
    assert calls == [("walk", "turn")]


def test_condition_compiler_aligns_world_route_masks_and_future_positions():
    ldf = make_ldf()
    state = ldf.init_stream_state(
        batch_size=1,
        window_tokens=6,
        initial_noise=make_initial_noise(),
        generator=torch.Generator().manual_seed(1),
    )
    state = replace(state, commit_index=1)
    text = TextTimeline("walk")
    text.update("turn", start_token=1)
    route = RoutePlan(
        times=np.asarray([0.0, 2.0]),
        points_xz=np.asarray([[10.0, 20.0], [12.0, 22.0]]),
        start_token=0,
        end_behavior=RouteEndBehavior.HOLD,
        version=4,
    )
    observations = RootObservationTimeline()
    observations.update(
        RootObservation(
            frame_index=4,
            value=np.asarray([10.0, 0.0, 20.0, 0.0, 1.0]),
            feature_mask=np.asarray([False, False, False, True, True]),
        )
    )
    compiler = InferenceConditionCompiler(
        text_embeddings=TextEmbeddingCache(encode_texts),
        root_mean=ldf.root_mean,
        root_std=ldf.root_std,
        active_tokens=ldf.chunk_size,
        max_horizon_token=2,
    )
    compiled = compiler.compile(
        state,
        text_timeline=text,
        route=route,
        route_revision=4,
        observations=observations,
        origin_xz=torch.tensor([10.0, 20.0]),
    )
    compiled.validate_for(state)
    condition = compiled.ldf_condition
    assert not condition.root_condition_mask[:, 0].any()
    assert condition.root_condition_mask[0, 1, 0].tolist() == [True, False, True, True, True]
    assert torch.allclose(
        condition.root_condition_value[0, 1, 0, [0, 2]],
        torch.tensor([0.2, 0.2]),
        atol=1e-6,
    )
    assert condition.future_timeline_position_ids.tolist() == [[3, 4]]
    assert len(condition.text_context) == 6
    assert not torch.equal(condition.text_context[0], condition.text_context[1])


def test_ldf_rebase_respects_clean_partial_and_noise_coefficients():
    ldf = make_ldf()
    state = ldf.init_stream_state(
        batch_size=1,
        window_tokens=6,
        initial_noise=make_initial_noise(root_x=1.0),
        generator=torch.Generator().manual_seed(2),
    )
    state = replace(
        state,
        current_step=4,
        commit_index=1,
        previous_root_frame=torch.tensor([[3.0, 0.0, 4.0, 1.0, 0.0]]),
    )
    rebased = ldf.rebase_stream_state(state, torch.tensor([2.0, 1.0]))
    assert torch.allclose(rebased.noisy_motion.root_motion[0, 0, :, 0], torch.full((4,), -1.0))
    assert torch.allclose(rebased.noisy_motion.root_motion[0, 1, :, 0], torch.zeros(4))
    assert torch.allclose(rebased.noisy_motion.root_motion[0, 2, :, 0], torch.ones(4))
    assert torch.allclose(
        rebased.previous_root_frame,
        torch.tensor([[1.0, 0.0, 3.0, 1.0, 0.0]]),
    )
    assert torch.equal(rebased.noisy_motion.latent_motion, state.noisy_motion.latent_motion)


def test_session_commits_one_token_decodes_four_frames_and_restores_snapshot():
    session = make_session()
    session.update_text("turn left")
    session.update_route(
        times=[0.0, 1.0],
        points_xz=[[0.0, 0.0], [1.0, 0.0]],
        reference=RouteReference.RELATIVE_TO_ACTOR,
    )
    first = session.generate_step()
    assert first.token_index == 0
    assert first.root_motion.shape == (1, 4, 5)
    assert first.body_prediction.continuous_body.shape == (1, 4, 261)
    assert first.committed_motion is not None
    assert first.committed_motion.token_length == 1
    assert session.commit_index == 1

    snapshot = session.create_snapshot()
    expected = session.generate_step()
    session.update_text("temporary")
    session.clear_route()
    session.restore_snapshot(snapshot)
    replayed = session.generate_step()
    assert torch.equal(replayed.root_motion, expected.root_motion)
    assert torch.equal(
        replayed.body_prediction.continuous_body,
        expected.body_prediction.continuous_body,
    )
    assert replayed.trace.text_revision == expected.trace.text_revision
    assert replayed.trace.route_revision == expected.trace.route_revision


def test_session_failure_discards_ldf_and_decoder_candidates():
    session = make_session()
    before = session.create_snapshot()

    def fail_decode(self, *args, **kwargs):
        raise RuntimeError("decode failed")

    session.body_vae.detokenize_step = types.MethodType(
        fail_decode, session.body_vae
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        session.generate_step()
    after = session.create_snapshot()
    assert session.commit_index == 0
    assert torch.equal(
        before.ldf_snapshot["noisy_root_motion"],
        after.ldf_snapshot["noisy_root_motion"],
    )
    assert all(
        torch.equal(left, right)
        for before_pair, after_pair in zip(
            before.decoder_state.caches, after.decoder_state.caches
        )
        for left, right in zip(before_pair, after_pair)
    )


def test_session_rebases_only_after_ldf_window_roll():
    session = make_session(root_x=2.0)
    chunks = list(session.generate(5))
    assert chunks[-1].trace.window_origin_after == 2
    assert chunks[-1].trace.rebased
    assert torch.allclose(session.origin_xz, torch.tensor([[12.0, 20.0]]))
    assert torch.allclose(
        session.ldf_state.previous_root_frame[:, [0, 2]], torch.zeros(1, 2)
    )


def test_fixed_stream_session_commits_without_rolling_the_window():
    session = make_session(root_x=2.0, rolling=False)
    chunks = list(session.generate(5))
    assert len(chunks) == 5
    assert session.ldf_state.window_origin == 0
    assert session.ldf_state.epoch == 0
    assert all(not chunk.trace.rebased for chunk in chunks)


def test_guidance_is_session_local_and_does_not_mutate_shared_model_defaults():
    guidance = GuidanceConfig(
        mode="separated", scale_text=2.5, scale_constraint=3.0
    )
    session = make_session(guidance=guidance)
    defaults = (
        session.ldf.cfg_scale_text,
        session.ldf.cfg_scale_constraint,
        session.ldf.cfg_scale_joint,
    )
    session.generate_step()
    assert defaults == (
        session.ldf.cfg_scale_text,
        session.ldf.cfg_scale_constraint,
        session.ldf.cfg_scale_joint,
    )
