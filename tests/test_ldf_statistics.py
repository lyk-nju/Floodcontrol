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


def test_root_statistics_reject_active_band_larger_than_parent_window():
    try:
        compute_root_statistics(
            _dataset(),
            min_frames=40,
            max_frames=40,
            random_yaw=False,
            active_tokens=11,
        )
    except ValueError as error:
        assert "active_tokens" in str(error)
    else:
        raise AssertionError("invalid root-statistics active band was accepted")


def test_root_statistics_mid_clip_parent_requires_nonzero_history(monkeypatch):
    class ScriptedRandom:
        def __init__(self):
            self.calls = []

        def randint(self, low, high):
            self.calls.append((low, high))
            return high if len(self.calls) % 2 else low

    sampler = ScriptedRandom()
    monkeypatch.setattr(
        "tools.compute_ldf_root_stats.random.Random",
        lambda _seed: sampler,
    )
    compute_root_statistics(
        _dataset(),
        min_frames=40,
        max_frames=48,
        windows_per_sample=1,
        random_yaw=False,
        active_tokens=5,
        seed=9,
    )

    source_calls = sampler.calls[0::2]
    history_calls = sampler.calls[1::2]
    assert all(high > 0 for _, high in source_calls)
    assert all(low == 1 for low, _ in history_calls)
