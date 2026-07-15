"""Bounded, lossless buffer for strict four-frame Web motion chunks."""

from __future__ import annotations

import threading
import time
from collections import deque

from .contracts import WebMotionChunk


class MotionChunkBuffer:
    """Coordinate one producer and one browser consumer without dropping data."""

    def __init__(self, *, target_chunks: int = 2, capacity_chunks: int = 8):
        target = int(target_chunks)
        capacity = int(capacity_chunks)
        if target <= 0:
            raise ValueError("target_chunks must be positive")
        if capacity < target:
            raise ValueError("capacity_chunks must be >= target_chunks")
        self.target_chunks = target
        self.capacity_chunks = capacity
        self._items: deque[WebMotionChunk] = deque()
        self._condition = threading.Condition()

    def put(
        self,
        chunk: WebMotionChunk,
        *,
        stop_event: threading.Event,
    ) -> bool:
        """Wait for capacity and enqueue a complete token transaction."""

        if not isinstance(chunk, WebMotionChunk):
            raise TypeError("chunk must be WebMotionChunk")
        with self._condition:
            while len(self._items) >= self.capacity_chunks and not stop_event.is_set():
                self._condition.wait(timeout=0.1)
            if stop_event.is_set():
                return False
            self._items.append(chunk)
            self._condition.notify_all()
            return True

    def get(self, *, timeout: float = 0.0) -> WebMotionChunk | None:
        """Return the oldest chunk, optionally waiting for the producer."""

        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            value = self._items.popleft()
            self._condition.notify_all()
            return value

    def wait_for_demand(self, stop_event: threading.Event) -> bool:
        """Block while the browser already has the configured prefetch target."""

        with self._condition:
            while len(self._items) >= self.target_chunks and not stop_event.is_set():
                self._condition.wait(timeout=0.1)
            return not stop_event.is_set()

    def clear(self) -> None:
        with self._condition:
            self._items.clear()
            self._condition.notify_all()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    @property
    def size(self) -> int:
        with self._condition:
            return len(self._items)


__all__ = ["MotionChunkBuffer"]
