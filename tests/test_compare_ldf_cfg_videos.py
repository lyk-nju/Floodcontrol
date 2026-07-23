from __future__ import annotations

import pytest

from tools.compare_ldf_cfg_videos import comparison_variants


def test_cfg_video_comparison_has_nocfg_and_requested_joint_scales():
    variants = comparison_variants((1.0, 2.0, 3.0))

    assert [(item.name, item.mode, item.joint_scale) for item in variants] == [
        ("nocfg", "nocfg", 1.0),
        ("joint_1p00", "joint", 1.0),
        ("joint_2p00", "joint", 2.0),
        ("joint_3p00", "joint", 3.0),
    ]


@pytest.mark.parametrize("scales", [(-1.0,), (float("nan"),), (1.0, 1.0)])
def test_cfg_video_comparison_rejects_invalid_or_duplicate_scales(scales):
    with pytest.raises(ValueError):
        comparison_variants(scales)
