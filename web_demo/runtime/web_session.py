"""Browser-session adapter over one authoritative ``InferenceSession``."""

from __future__ import annotations

import threading
import time

import numpy as np

from utils.inference import GuidanceConfig, InferenceSession
from utils.inference.geometry import assign_times_by_arclength

from .chunk_buffer import MotionChunkBuffer
from .contracts import WebMotionChunk, WebSessionState


class WebSession:
    """Serialize controls and generation for one non-thread-safe model session."""

    def __init__(
        self,
        *,
        session_id: str,
        inference: InferenceSession,
        execution_lock: threading.Lock,
        target_chunks: int,
        capacity_chunks: int,
    ):
        if not str(session_id):
            raise ValueError("session_id must be non-empty")
        self.session_id = str(session_id)
        self.inference = inference
        self.buffer = MotionChunkBuffer(
            target_chunks=target_chunks,
            capacity_chunks=capacity_chunks,
        )
        self._execution_lock = execution_lock
        self._inference_lock = threading.RLock()
        self._control = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = WebSessionState.IDLE
        self._error: str | None = None
        self._session_epoch = 0
        self._last_consumed_at = time.monotonic()
        self._last_chunk: WebMotionChunk | None = None

    @property
    def state(self) -> WebSessionState:
        with self._control:
            return self._state

    @property
    def error(self) -> str | None:
        with self._control:
            return self._error

    @property
    def session_epoch(self) -> int:
        with self._control:
            return self._session_epoch

    @property
    def inactive_seconds(self) -> float:
        with self._control:
            return max(0.0, time.monotonic() - self._last_consumed_at)

    def start(self) -> None:
        with self._control:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("session generation worker is already running")
            self._stop_event.clear()
            self._error = None
            self._state = WebSessionState.RUNNING
            self._last_consumed_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._run,
                name=f"floodcontrol-web-{self.session_id}",
                daemon=True,
            )
            self._thread.start()

    def _wait_until_running(self) -> bool:
        with self._control:
            while self._state is WebSessionState.PAUSED and not self._stop_event.is_set():
                self._control.wait(timeout=0.1)
            return self._state is WebSessionState.RUNNING and not self._stop_event.is_set()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                if not self._wait_until_running():
                    if self._stop_event.is_set():
                        break
                    continue
                if not self.buffer.wait_for_demand(self._stop_event):
                    break
                with self._inference_lock:
                    with self._execution_lock:
                        generated = self.inference.generate_step()
                        chunk = WebMotionChunk.from_generated(
                            generated,
                            session_epoch=self.session_epoch,
                        )
                if not self.buffer.put(chunk, stop_event=self._stop_event):
                    break
                with self._control:
                    self._last_chunk = chunk
        except Exception as exc:  # Background errors must become observable API state.
            with self._control:
                self._error = f"{type(exc).__name__}: {exc}"
                self._state = WebSessionState.ERROR
                self._control.notify_all()
            self.buffer.wake_all()
        finally:
            with self._control:
                if self._state is WebSessionState.RESETTING:
                    self._state = WebSessionState.IDLE
                self._control.notify_all()

    def pause(self) -> None:
        with self._control:
            if self._state is not WebSessionState.RUNNING:
                raise RuntimeError("only a running session can be paused")
            self._state = WebSessionState.PAUSED

    def resume(self) -> None:
        with self._control:
            if self._state is not WebSessionState.PAUSED:
                raise RuntimeError("only a paused session can be resumed")
            self._state = WebSessionState.RUNNING
            self._control.notify_all()

    def stop(self, *, timeout: float = 5.0) -> None:
        with self._control:
            thread = self._thread
            if thread is None:
                self._state = WebSessionState.IDLE
                self.buffer.clear()
                return
            if self._state is not WebSessionState.ERROR:
                self._state = WebSessionState.RESETTING
            self._stop_event.set()
            self._session_epoch += 1
            self._control.notify_all()
        self.buffer.wake_all()
        thread.join(timeout=max(0.0, float(timeout)))
        if thread.is_alive():
            with self._control:
                self._state = WebSessionState.ERROR
                self._error = "generation worker did not stop before timeout"
            raise RuntimeError(self._error)
        with self._control:
            self._thread = None
            if self._state is not WebSessionState.ERROR:
                self._state = WebSessionState.IDLE
        self.buffer.clear()

    def pop_chunk(self, *, timeout: float) -> WebMotionChunk | None:
        self.touch_consumption()
        return self.buffer.get(timeout=timeout)

    def touch_consumption(self) -> None:
        with self._control:
            self._last_consumed_at = time.monotonic()

    def update_text(self, text: str) -> dict:
        with self._inference_lock:
            self.inference.update_text(str(text))
            return {
                "effective_token": int(self.inference.commit_index),
                "revision": int(self.inference.text_timeline.revision),
                "text": str(text),
            }

    def update_route(
        self,
        *,
        points_xz,
        duration_seconds: float,
        reference: str,
        end_behavior: str,
        source: str,
    ) -> dict:
        points = np.asarray(points_xz, dtype=np.float32)
        times = assign_times_by_arclength(points, duration_seconds)
        with self._inference_lock:
            route = self.inference.update_route(
                times=times,
                points_xz=points,
                reference=reference,
                end_behavior=end_behavior,
                source=source,
            )
        return {
            "version": route.version,
            "start_token": route.start_token,
            "times": route.times.tolist(),
            "points_xz": route.points_xz.tolist(),
            "end_behavior": route.end_behavior.value,
            "source": route.source,
        }

    def clear_route(self) -> dict:
        with self._inference_lock:
            self.inference.clear_route()
            return {
                "version": int(self.inference.route_revision),
                "start_token": int(self.inference.commit_index),
            }

    def update_guidance(self, guidance: GuidanceConfig) -> dict:
        with self._inference_lock:
            self.inference.update_guidance(guidance)
        return {
            "mode": guidance.mode,
            "scale_text": guidance.scale_text,
            "scale_constraint": guidance.scale_constraint,
            "scale_joint": guidance.scale_joint,
        }

    def status(self) -> dict:
        with self._inference_lock:
            commit_index = int(self.inference.commit_index)
            ldf_state = self.inference.ldf_state
            window_origin = int(ldf_state.window_origin)
            window_epoch = int(ldf_state.epoch)
            text_revision = int(self.inference.text_timeline.revision)
            route_revision = int(self.inference.route_revision)
            route = self.inference.route
        with self._control:
            payload = {
                "session_id": self.session_id,
                "session_epoch": self._session_epoch,
                "state": self._state.value,
                "error": self._error,
                "buffered_chunks": self.buffer.size,
                "buffer_target_chunks": self.buffer.target_chunks,
                "buffer_capacity_chunks": self.buffer.capacity_chunks,
                "commit_index": commit_index,
                "window_origin": window_origin,
                "window_epoch": window_epoch,
                "text_revision": text_revision,
                "route_revision": route_revision,
                "route": None,
                "last_trace": None if self._last_chunk is None else self._last_chunk.trace,
            }
        if route is not None:
            payload["route"] = {
                "version": route.version,
                "start_token": route.start_token,
                "points_xz": route.points_xz.tolist(),
                "times": route.times.tolist(),
                "end_behavior": route.end_behavior.value,
                "source": route.source,
            }
        return payload


__all__ = ["WebSession"]
