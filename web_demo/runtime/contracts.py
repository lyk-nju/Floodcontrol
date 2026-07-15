"""Typed process-boundary contracts for the Floodcontrol Web runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
import torch

from utils.inference import GeneratedMotionChunk
from utils.motion_process import NUM_JOINTS, ROOT_DIM, recover_joint_positions
from utils.token_frame import FRAMES_PER_TOKEN


class WebSessionState(str, Enum):
    """Lifecycle state of one browser-owned inference session."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    RESETTING = "resetting"
    ERROR = "error"


@dataclass(frozen=True)
class WebMotionChunk:
    """One atomic model commit prepared for JSON transport.

    The Web layer never splits model state at a frame boundary. One object is
    always exactly one latent token and therefore four consecutive frames.
    """

    session_epoch: int
    token_index: int
    frame_start: int
    root_motion: np.ndarray
    joint_positions: np.ndarray
    contact_probability: np.ndarray
    trace: dict

    @classmethod
    def from_generated(
        cls,
        generated: GeneratedMotionChunk,
        *,
        session_epoch: int,
    ) -> "WebMotionChunk":
        """Synchronize a generated chunk to CPU and recover display joints."""

        generated.validate()
        root = generated.root_motion.detach()
        body = generated.body_prediction.continuous_body.detach()
        joints = recover_joint_positions(root, body)
        contacts = generated.body_prediction.contact_logits.detach().sigmoid()
        value = cls(
            session_epoch=int(session_epoch),
            token_index=int(generated.token_index),
            frame_start=int(generated.token_index) * FRAMES_PER_TOKEN,
            root_motion=root[0].to(device="cpu", dtype=torch.float32).numpy().copy(),
            joint_positions=joints[0]
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .copy(),
            contact_probability=contacts[0]
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .copy(),
            trace=asdict(generated.trace),
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.session_epoch < 0 or self.token_index < 0 or self.frame_start < 0:
            raise ValueError("chunk indices must be non-negative")
        if self.frame_start != self.token_index * FRAMES_PER_TOKEN:
            raise ValueError("frame_start must match the strict four-frame token contract")
        expected = {
            "root_motion": (FRAMES_PER_TOKEN, ROOT_DIM),
            "joint_positions": (FRAMES_PER_TOKEN, NUM_JOINTS, 3),
            "contact_probability": (FRAMES_PER_TOKEN, 4),
        }
        for name, shape in expected.items():
            array = np.asarray(getattr(self, name))
            if tuple(array.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if not bool(np.isfinite(array).all()):
                raise ValueError(f"{name} contains non-finite values")

    def to_payload(self) -> dict:
        """Return a JSON-compatible four-frame payload."""

        frames = []
        for offset in range(FRAMES_PER_TOKEN):
            frames.append(
                {
                    "frame_index": self.frame_start + offset,
                    "root_motion": self.root_motion[offset].tolist(),
                    "joints": self.joint_positions[offset].tolist(),
                    "contact_probability": self.contact_probability[offset].tolist(),
                }
            )
        return {
            "session_epoch": self.session_epoch,
            "token_index": self.token_index,
            "frame_start": self.frame_start,
            "frames": frames,
            "trace": dict(self.trace),
        }


__all__ = ["WebMotionChunk", "WebSessionState"]
