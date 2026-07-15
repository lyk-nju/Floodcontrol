"""Authoritative Web facade over shared models and one active session."""

from __future__ import annotations

import threading
import uuid
from typing import Callable

from utils.inference import GuidanceConfig

from web_demo.config import WebConfig

from .model_bundle import ModelBundle
from .model_loader import load_model_bundle
from .web_session import WebSession


class WebRuntimeError(RuntimeError):
    """Base class for user-visible runtime state failures."""


class SessionConflictError(WebRuntimeError):
    def __init__(self, active_session_id: str):
        super().__init__("another browser session is already generating")
        self.active_session_id = active_session_id


class SessionNotFoundError(WebRuntimeError):
    pass


class WebRuntime:
    """Own lazy model loading, active-session policy and GPU serialization."""

    def __init__(
        self,
        config: WebConfig,
        *,
        bundle_loader: Callable[[WebConfig], ModelBundle] = load_model_bundle,
        inference_factory: Callable | None = None,
        start_monitor: bool = True,
    ):
        self.config = config
        self._bundle_loader = bundle_loader
        self._inference_factory = inference_factory
        self._bundle: ModelBundle | None = None
        self._active: WebSession | None = None
        self._operation_lock = threading.RLock()
        self._execution_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        if start_monitor:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="floodcontrol-web-session-monitor",
                daemon=True,
            )
            self._monitor_thread.start()

    def _load_bundle(self) -> ModelBundle:
        with self._operation_lock:
            if self._bundle is None:
                self._bundle = self._bundle_loader(self.config)
            return self._bundle

    def _require_session(self, session_id: str) -> WebSession:
        value = str(session_id)
        with self._operation_lock:
            if self._active is None or self._active.session_id != value:
                raise SessionNotFoundError("session is not the active Web session")
            return self._active

    def start_session(
        self,
        *,
        text: str,
        seed: int,
        initial_world_xz,
        initial_yaw: float | None,
        force: bool,
        guidance: GuidanceConfig | None = None,
        initial_route: dict | None = None,
    ) -> dict:
        with self._operation_lock:
            if self._active is not None:
                if not force:
                    raise SessionConflictError(self._active.session_id)
                self._active.stop(timeout=self.config.worker_stop_timeout_seconds)
                self._active = None

            bundle = self._load_bundle()
            selected_guidance = guidance or self.config.guidance
            if self._inference_factory is None:
                inference = bundle.create_session(
                    config=self.config.inference,
                    guidance=selected_guidance,
                    seed=seed,
                    initial_world_xz=initial_world_xz,
                    initial_yaw=initial_yaw,
                    initial_text=text,
                )
            else:
                inference = self._inference_factory(
                    bundle=bundle,
                    config=self.config.inference,
                    guidance=selected_guidance,
                    seed=seed,
                    initial_world_xz=initial_world_xz,
                    initial_yaw=initial_yaw,
                    initial_text=text,
                )
            session = WebSession(
                session_id=uuid.uuid4().hex,
                inference=inference,
                execution_lock=self._execution_lock,
                target_chunks=self.config.buffer_target_chunks,
                capacity_chunks=self.config.buffer_capacity_chunks,
            )
            self._active = session
            try:
                if initial_route is not None:
                    session.update_route(**initial_route)
                session.start()
            except Exception:
                self._active = None
                session.stop(timeout=self.config.worker_stop_timeout_seconds)
                raise
            return session.status()

    def update_text(self, session_id: str, text: str) -> dict:
        with self._operation_lock:
            return self._require_session(session_id).update_text(text)

    def update_route(self, session_id: str, **route) -> dict:
        with self._operation_lock:
            return self._require_session(session_id).update_route(**route)

    def clear_route(self, session_id: str) -> dict:
        with self._operation_lock:
            return self._require_session(session_id).clear_route()

    def update_guidance(self, session_id: str, guidance: GuidanceConfig) -> dict:
        with self._operation_lock:
            return self._require_session(session_id).update_guidance(guidance)

    def pause(self, session_id: str) -> dict:
        with self._operation_lock:
            session = self._require_session(session_id)
            session.pause()
            return session.status()

    def resume(self, session_id: str) -> dict:
        with self._operation_lock:
            session = self._require_session(session_id)
            session.resume()
            return session.status()

    def pop_chunk(self, session_id: str, *, timeout: float):
        return self._require_session(session_id).pop_chunk(timeout=timeout)

    def reset(self, session_id: str) -> None:
        with self._operation_lock:
            session = self._require_session(session_id)
            session.stop(timeout=self.config.worker_stop_timeout_seconds)
            if self._active is session:
                self._active = None

    def status(self, session_id: str | None = None) -> dict:
        with self._operation_lock:
            active = self._active
            base = {
                "initialized": self._bundle is not None,
                "runtime_status": self.config.status,
                "runtime_message": self.config.message,
                "active_session_id": None if active is None else active.session_id,
            }
            if session_id is None:
                if active is not None:
                    base["session"] = active.status()
                return base
            session = self._require_session(session_id)
            base["session"] = session.status()
            return base

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self.config.monitor_interval_seconds):
            with self._operation_lock:
                session = self._active
                if session is None:
                    continue
                if session.state.value != "running":
                    continue
                if session.inactive_seconds <= self.config.consumption_timeout_seconds:
                    continue
                stopped = False
                try:
                    session.stop(timeout=self.config.worker_stop_timeout_seconds)
                    stopped = True
                except RuntimeError:
                    pass
                if stopped and self._active is session:
                    self._active = None

    def shutdown(self) -> None:
        self._monitor_stop.set()
        with self._operation_lock:
            active = self._active
            if active is not None:
                try:
                    active.stop(timeout=self.config.worker_stop_timeout_seconds)
                finally:
                    self._active = None
        thread = self._monitor_thread
        if thread is not None:
            thread.join(timeout=self.config.worker_stop_timeout_seconds)


__all__ = [
    "SessionConflictError",
    "SessionNotFoundError",
    "WebRuntime",
    "WebRuntimeError",
]
