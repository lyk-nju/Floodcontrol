from pathlib import Path

import numpy as np
import pytest
import torch

import utils.visualization as visualization
import utils.visualization.motion_video as motion_video


ROOT = Path(__file__).resolve().parents[1]


class RecordingWriter:
    def __init__(self, *, fail_on_append: bool = False):
        self.frames: list[np.ndarray] = []
        self.closed = False
        self.fail_on_append = fail_on_append

    def append_data(self, frame: np.ndarray) -> None:
        if self.fail_on_append:
            raise RuntimeError("encoder failed")
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


def _joints(frames: int = 2) -> torch.Tensor:
    joints = torch.zeros(frames, 22, 3)
    joints[..., 0] = torch.arange(22) * 0.02
    joints[..., 1] = torch.arange(22) * 0.04
    joints[:, :, 2] = torch.arange(frames)[:, None] * 0.01
    return joints


def test_visualization_public_api_is_root5_body259_only():
    assert len(visualization.HUMANML22_CHAINS) == 5
    assert max(max(chain) for chain in visualization.HUMANML22_CHAINS) == 21
    assert not (ROOT / "utils" / "visualization" / "video.py").exists()
    for old_name in (
        "get_humanml3d_chains",
        "render_simple_skeleton_video",
        "render_skeleton_video",
        "render_root_trajectory_video",
        "render_single_video",
        "render_video",
        "make_composite_compare_videos",
    ):
        assert not hasattr(visualization, old_name)


def test_joint_renderer_writes_one_frame_per_motion_frame(tmp_path, monkeypatch):
    writer = RecordingWriter()
    monkeypatch.setattr(
        motion_video.imageio,
        "get_writer",
        lambda path, fps: writer,
    )
    output = tmp_path / "nested" / "motion.mp4"
    motion_video.render_joint_video(_joints(3), output, fps=20)
    assert output.parent.is_dir()
    assert len(writer.frames) == 3
    assert writer.frames[0].shape == (480, 480, 3)
    assert writer.frames[0].dtype == np.uint8
    assert writer.closed


def test_joint_renderer_embeds_target_and_generated_trajectories_in_scene(
    tmp_path,
    monkeypatch,
):
    writer = RecordingWriter()
    monkeypatch.setattr(
        motion_video.imageio,
        "get_writer",
        lambda path, fps: writer,
    )
    joints = _joints(3)
    joints[..., 0] += torch.linspace(0.0, 1.0, 3)[:, None]
    target_xz = torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    motion_video.render_joint_video(
        joints,
        tmp_path / "motion_with_route.mp4",
        traj_xz=target_xz,
        traj_mask=torch.ones(3, dtype=torch.bool),
        show_full_trajectory=True,
        show_generated_trajectory=True,
    )

    final_frame = writer.frames[-1]
    scene = final_frame[30:]
    assert np.any(
        np.all(scene == motion_video.TARGET_TRAJECTORY_COLOR, axis=-1)
    )
    assert np.any(
        np.all(scene == motion_video.GENERATED_TRAJECTORY_COLOR, axis=-1)
    )


def test_joint_renderer_closes_writer_when_encoding_fails(tmp_path, monkeypatch):
    writer = RecordingWriter(fail_on_append=True)
    monkeypatch.setattr(
        motion_video.imageio,
        "get_writer",
        lambda path, fps: writer,
    )
    with pytest.raises(RuntimeError, match="encoder failed"):
        motion_video.render_joint_video(_joints(1), tmp_path / "motion.mp4")
    assert writer.closed


@pytest.mark.parametrize(
    ("joint_positions", "message"),
    [
        (torch.zeros(0, 22, 3), "at least one frame"),
        (torch.zeros(2, 21, 3), r"must be \[F,22,3\]"),
        (torch.full((2, 22, 3), float("nan")), "non-finite"),
    ],
)
def test_joint_renderer_rejects_invalid_motion(joint_positions, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        motion_video.render_joint_video(joint_positions, tmp_path / "motion.mp4")


def test_motion_renderer_recovers_world_joints(monkeypatch, tmp_path):
    captured = {}

    def record(joints, output_path, **kwargs):
        captured["joints"] = joints
        captured["output_path"] = output_path
        captured["kwargs"] = kwargs

    monkeypatch.setattr(motion_video, "render_joint_video", record)
    root = torch.tensor(
        [[2.0, 1.2, -3.0, 1.0, 0.0], [2.1, 1.2, -3.0, 1.0, 0.0]]
    )
    body = torch.zeros(2, 259)
    output = tmp_path / "motion.mp4"
    motion_video.render_motion_video(root, body, output, fps=24)
    assert captured["joints"].shape == (2, 22, 3)
    assert torch.equal(captured["joints"][:, 0], root[:, :3])
    assert captured["output_path"] == output
    assert captured["kwargs"]["fps"] == 24
