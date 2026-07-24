from __future__ import annotations

import pytest
import torch

from tools.sweep_ldf_t2m_cfg import (
    GuidanceVariant,
    assign_variant_groups,
    guidance_variants,
)
from utils.training.ldf.evaluation.t2m import (
    build_t2m_evaluation_batches,
    shard_t2m_evaluation_batches,
)


def test_t2m_cfg_sweep_has_nocfg_and_requested_joint_scales():
    variants = guidance_variants((1.0, 1.5, 2.0))

    assert variants == (
        GuidanceVariant(name="nocfg", mode="nocfg", joint_scale=1.0),
        GuidanceVariant(name="joint_1p00", mode="joint", joint_scale=1.0),
        GuidanceVariant(name="joint_1p50", mode="joint", joint_scale=1.5),
        GuidanceVariant(name="joint_2p00", mode="joint", joint_scale=2.0),
    )


@pytest.mark.parametrize(
    "scales",
    [
        (-1.0,),
        (float("nan"),),
        (1.0, 1.0),
    ],
)
def test_t2m_cfg_sweep_rejects_invalid_or_duplicate_scales(scales):
    with pytest.raises(ValueError):
        guidance_variants(scales)


def test_t2m_cfg_sweep_assigns_complete_variants_across_gpus():
    variants = guidance_variants((1.0, 1.5, 2.0, 2.5, 3.0))

    groups = assign_variant_groups(variants, (0, 2, 3, 5))

    assert groups == (
        (0, (variants[0], variants[4])),
        (2, (variants[1], variants[5])),
        (3, (variants[2],)),
        (5, (variants[3],)),
    )
    flattened = [variant for _, group in groups for variant in group]
    assert sorted(flattened, key=lambda item: item.name) == sorted(
        variants,
        key=lambda item: item.name,
    )


@pytest.mark.parametrize("devices", [(), (0, 0), (-1,)])
def test_t2m_cfg_sweep_rejects_invalid_device_assignments(devices):
    with pytest.raises(ValueError):
        assign_variant_groups(guidance_variants((1.0,)), devices)


def test_t2m_batches_are_fixed_before_world_size_sharding():
    samples = []
    for index, frames in enumerate((8, 12, 8, 16, 12, 8, 16)):
        samples.append(
            (
                index,
                {
                    "root_motion": torch.zeros(frames, 5),
                    "name": f"sample_{index}",
                },
            )
        )
    batches = build_t2m_evaluation_batches(
        samples,
        maximum_frames=16,
        batch_size=2,
    )
    canonical = [
        tuple(index for index, _ in batch.samples)
        for batch in batches
    ]

    for world_size in (1, 4, 8):
        observed = []
        for rank in range(world_size):
            shard = shard_t2m_evaluation_batches(
                batches,
                rank=rank,
                world_size=world_size,
            )
            observed.extend(
                tuple(index for index, _ in batch.samples)
                for batch in shard
            )
        assert sorted(observed) == sorted(canonical)
