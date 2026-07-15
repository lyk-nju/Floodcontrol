import torch

from tools.compute_ldf_root_stats import compute_root_statistics


def _dataset():
    samples = []
    for index in range(3):
        root = torch.zeros(64, 5)
        root[:, 0] = torch.arange(64, dtype=torch.float32) + index * 10
        root[:, 1] = 1.0 + index
        root[:, 2] = torch.arange(64, dtype=torch.float32) * 0.5
        root[:, 3] = 1.0
        samples.append({"root_motion": root})
    return samples


def test_root_statistics_are_deterministic_and_follow_window_rebase():
    first = compute_root_statistics(
        _dataset(),
        min_frames=40,
        max_frames=48,
        windows_per_sample=2,
        random_yaw=True,
        seed=7,
    )
    second = compute_root_statistics(
        _dataset(),
        min_frames=40,
        max_frames=48,
        windows_per_sample=2,
        random_yaw=True,
        seed=7,
    )
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first[0].shape == (5,)
    assert first[1].shape == (5,)
    assert torch.isfinite(first[0]).all()
    assert (first[1] > 0).all()


def test_root_statistics_reject_non_aligned_training_windows():
    try:
        compute_root_statistics(
            _dataset(), min_frames=38, max_frames=40, random_yaw=False
        )
    except ValueError as error:
        assert "multiple of four" in str(error)
    else:
        raise AssertionError("non-aligned root statistics window was accepted")
