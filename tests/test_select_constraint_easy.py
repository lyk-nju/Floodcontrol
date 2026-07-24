from __future__ import annotations

import math
import json
import sys

import numpy as np

from tools.select_constraint_easy import (
    ConstraintEasyThresholds,
    compute_trajectory_metrics,
    constraint_easy_score,
    is_constraint_easy,
    main,
)


def _root_from_xz(xz: np.ndarray) -> np.ndarray:
    root = np.zeros((len(xz), 5), dtype=np.float32)
    root[:, [0, 2]] = xz
    root[:, 1] = 1.0
    root[:, 3] = 1.0
    return root


def test_stationary_and_small_local_motion_are_constraint_easy():
    thresholds = ConstraintEasyThresholds()
    stationary = compute_trajectory_metrics(
        _root_from_xz(np.zeros((40, 2), dtype=np.float32))
    )
    local = compute_trajectory_metrics(
        _root_from_xz(
            np.stack(
                (
                    np.linspace(0.0, 0.2, 40),
                    0.02 * np.sin(np.linspace(0.0, 2.0 * math.pi, 40)),
                ),
                axis=-1,
            ).astype(np.float32)
        )
    )

    assert is_constraint_easy(stationary, thresholds)
    assert is_constraint_easy(local, thresholds)
    assert constraint_easy_score(stationary, thresholds) == 0.0


def test_long_straight_and_closed_circle_are_not_constraint_easy():
    thresholds = ConstraintEasyThresholds()
    straight = compute_trajectory_metrics(
        _root_from_xz(
            np.stack(
                (np.linspace(0.0, 2.0, 80), np.zeros(80)),
                axis=-1,
            ).astype(np.float32)
        )
    )
    angles = np.linspace(0.0, 2.0 * math.pi, 160)
    circle = compute_trajectory_metrics(
        _root_from_xz(
            np.stack((np.cos(angles), np.sin(angles)), axis=-1).astype(np.float32)
        )
    )

    assert not is_constraint_easy(straight, thresholds)
    assert not is_constraint_easy(circle, thresholds)
    assert circle.displacement_m < 1e-4
    assert circle.spatial_extent_m > thresholds.max_spatial_extent_m


def test_constraint_easy_metrics_are_translation_and_yaw_invariant():
    xz = np.stack(
        (np.linspace(0.0, 0.3, 40), np.linspace(0.0, 0.1, 40)),
        axis=-1,
    ).astype(np.float32)
    angle = math.radians(67.0)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    transformed = xz @ rotation.T + np.asarray([17.0, -3.0], dtype=np.float32)

    original_metrics = compute_trajectory_metrics(_root_from_xz(xz))
    transformed_metrics = compute_trajectory_metrics(_root_from_xz(transformed))

    for name in (
        "path_length_m",
        "displacement_m",
        "spatial_extent_m",
        "max_radius_from_start_m",
        "mean_speed_mps",
        "p95_speed_mps",
        "moving_ratio",
    ):
        assert math.isclose(
            getattr(original_metrics, name),
            getattr(transformed_metrics, name),
            rel_tol=1e-5,
            abs_tol=1e-5,
        )


def test_selector_writes_original_order_split_and_audit_report(
    tmp_path,
    monkeypatch,
):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    stationary = _root_from_xz(np.zeros((40, 2), dtype=np.float32))
    moving = _root_from_xz(
        np.stack(
            (np.linspace(0.0, 2.0, 80), np.zeros(80)),
            axis=-1,
        ).astype(np.float32)
    )
    np.savez(
        artifact_dir / "stationary.npz",
        root_motion=stationary,
    )
    np.savez(
        artifact_dir / "moving.npz",
        root_motion=moving,
    )
    (tmp_path / "train.txt").write_text("moving\nstationary\n")
    output = tmp_path / "train_constraint_easy.txt"
    summary = tmp_path / "constraint_easy.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_constraint_easy",
            "--meta",
            str(tmp_path / "train.txt"),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--workers",
            "1",
        ],
    )

    main()

    assert output.read_text() == "stationary\n"
    report = json.loads(summary.read_text())
    assert report["contract"] == "constraint_easy_v1"
    assert report["counts"] == {
        "constraint_easy": 1,
        "total": 2,
        "trajectory_teaching": 1,
    }
