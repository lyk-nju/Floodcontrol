import threading
from types import SimpleNamespace

import numpy as np
import torch

from utils.conditions.vae import BodyPrediction
from utils.inference import (
    GeneratedMotionChunk,
    GuidanceConfig,
    InferenceConfig,
    InferenceStepTrace,
    RouteEndBehavior,
    RoutePlan,
    RouteReference,
)
from web_demo.app import create_app
from web_demo.config import WebConfig
from web_demo.runtime.chunk_buffer import MotionChunkBuffer
from web_demo.runtime.contracts import WebMotionChunk
from web_demo.runtime.web_runtime import WebRuntime


class FakeInferenceSession:
    def __init__(self):
        self.commit_index = 0
        self.ldf_state = SimpleNamespace(window_origin=0, epoch=0)
        self.text_timeline = SimpleNamespace(revision=0)
        self.route_revision = 0
        self.route = None
        self.guidance = GuidanceConfig()

    def generate_step(self):
        token = self.commit_index
        root = torch.zeros(1, 4, 5)
        root[..., 0] = torch.arange(4) * 0.01 + token * 0.04
        root[..., 1] = 1.0
        root[..., 3] = 1.0
        body = BodyPrediction(
            continuous_body=torch.zeros(1, 4, 261),
            contact_logits=torch.zeros(1, 4, 4),
        )
        trace = InferenceStepTrace(
            token_index=token,
            window_origin_before=self.ldf_state.window_origin,
            window_origin_after=self.ldf_state.window_origin,
            window_epoch_before=self.ldf_state.epoch,
            window_epoch_after=self.ldf_state.epoch,
            text_revision=self.text_timeline.revision,
            route_revision=self.route_revision,
            observation_revision=0,
            rebased=False,
        )
        self.commit_index += 1
        return GeneratedMotionChunk(token, root, body, trace)

    def update_text(self, _text):
        self.text_timeline.revision += 1

    def update_route(
        self, *, times, points_xz, reference, end_behavior, source
    ):
        self.route_revision += 1
        route = RoutePlan(
            times=np.asarray(times),
            points_xz=np.asarray(points_xz),
            start_token=self.commit_index,
            end_behavior=RouteEndBehavior(end_behavior),
            version=self.route_revision,
            source=source,
        )
        if RouteReference(reference) is RouteReference.RELATIVE_TO_ACTOR:
            route = route.resolve_world(reference, np.array([1.0, 2.0]))
        self.route = route
        return route

    def clear_route(self):
        self.route = None
        self.route_revision += 1

    def update_guidance(self, guidance):
        self.guidance = guidance


def make_config(**overrides):
    values = dict(
        status="test",
        message="",
        inference=InferenceConfig(window_tokens=4, future_constraint_tokens=2),
        guidance=GuidanceConfig(),
        buffer_target_chunks=1,
        buffer_capacity_chunks=2,
        consumption_timeout_seconds=30.0,
        monitor_interval_seconds=1.0,
    )
    values.update(overrides)
    return WebConfig(**values)


def make_runtime():
    return WebRuntime(
        make_config(),
        bundle_loader=lambda _config: object(),
        inference_factory=lambda **_kwargs: FakeInferenceSession(),
        start_monitor=False,
    )


def fake_web_chunk(token_index=0):
    generated = FakeInferenceSession().generate_step()
    if token_index:
        generated = GeneratedMotionChunk(
            token_index,
            generated.root_motion,
            generated.body_prediction,
            generated.trace,
        )
    return WebMotionChunk.from_generated(generated, session_epoch=0)


def test_web_motion_chunk_is_exactly_four_frames():
    chunk = fake_web_chunk()
    payload = chunk.to_payload()
    assert len(payload["frames"]) == 4
    assert chunk.root_motion.shape == (4, 5)
    assert chunk.joint_positions.shape == (4, 22, 3)
    assert [frame["frame_index"] for frame in payload["frames"]] == [0, 1, 2, 3]


def test_chunk_buffer_uses_backpressure_instead_of_silent_drop():
    buffer = MotionChunkBuffer(target_chunks=1, capacity_chunks=1)
    stop = threading.Event()
    first = fake_web_chunk()
    second = fake_web_chunk(1)
    assert buffer.put(first, stop_event=stop)
    inserted = threading.Event()

    def put_second():
        if buffer.put(second, stop_event=stop):
            inserted.set()

    thread = threading.Thread(target=put_second)
    thread.start()
    assert not inserted.wait(0.05)
    assert buffer.get(timeout=0.0).token_index == 0
    assert inserted.wait(1.0)
    assert buffer.get(timeout=0.0).token_index == 1
    stop.set()
    buffer.wake_all()
    thread.join(timeout=1.0)


def test_web_runtime_serializes_hybrid_session_and_route_updates():
    runtime = make_runtime()
    session = runtime.start_session(
        text="walk",
        seed=3,
        initial_world_xz=(0.0, 0.0),
        initial_yaw=None,
        force=False,
        initial_route=None,
    )
    session_id = session["session_id"]
    chunk = runtime.pop_chunk(session_id, timeout=1.0)
    assert chunk is not None
    assert len(chunk.to_payload()["frames"]) == 4
    text = runtime.update_text(session_id, "turn left")
    assert text["revision"] == 1
    route = runtime.update_route(
        session_id,
        points_xz=[[0.0, 0.0], [1.0, 0.0]],
        duration_seconds=2.0,
        reference="relative_to_actor",
        end_behavior="hold",
        source="test",
    )
    assert route["points_xz"] == [[1.0, 2.0], [2.0, 2.0]]
    runtime.pause(session_id)
    assert runtime.status(session_id)["session"]["state"] == "paused"
    runtime.resume(session_id)
    runtime.reset(session_id)
    assert runtime.status()["active_session_id"] is None
    runtime.shutdown()


def test_web_api_uses_session_urls_and_returns_atomic_chunks():
    runtime = make_runtime()
    app = create_app("configs/stream.yaml", runtime=runtime)
    client = app.test_client()
    start = client.post(
        "/api/sessions",
        json={
            "text": "walk",
            "seed": 0,
            "route": {
                "points_xz": [[0, 0], [1, 0]],
                "duration_seconds": 2,
                "reference": "world",
                "end_behavior": "hold",
            },
        },
    )
    assert start.status_code == 201
    session_id = start.get_json()["session"]["session_id"]
    assert start.get_json()["session"]["route"]["start_token"] == 0

    conflict = client.post("/api/sessions", json={"text": "run"})
    assert conflict.status_code == 409
    assert conflict.get_json()["conflict"] is True

    route = client.put(
        f"/api/sessions/{session_id}/route",
        json={
            "points_xz": [[0, 0], [1, 0]],
            "duration_seconds": 2,
            "reference": "world",
            "end_behavior": "release",
        },
    )
    assert route.status_code == 200
    response = client.get(
        f"/api/sessions/{session_id}/chunks/next?wait_ms=1000"
    )
    assert response.status_code == 200
    assert len(response.get_json()["chunk"]["frames"]) == 4
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    runtime.shutdown()


def test_real_web_loader_reports_checkpoint_blocker_over_http():
    app = create_app("configs/stream.yaml")
    client = app.test_client()
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.get_json()["runtime_status"] == "BLOCKED_ON_LDF_CHECKPOINT"
    start = client.post("/api/sessions", json={"text": "walk"})
    assert start.status_code == 503
    assert "BLOCKED_ON_LDF_CHECKPOINT" in start.get_json()["message"]
    app.extensions["floodcontrol_runtime"].shutdown()
