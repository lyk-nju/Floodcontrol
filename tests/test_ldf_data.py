import pytest
import torch
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from utils.training.ldf.data import (
    DistributedShardSampler,
    LDFSpanCollator,
    LengthBucketBatchSampler,
    MinimumFrameDataset,
    ResumableDataLoader,
    _mix_seed,
    create_dataloaders,
)


def make_sample(frames=40, name="sample"):
    root = torch.zeros(frames, 5)
    root[:, 0] = torch.arange(frames, dtype=torch.float32)
    root[:, 2] = torch.arange(frames, dtype=torch.float32) * 2
    root[:, 3] = 1
    body = torch.zeros(frames, 265)
    body[:, 0] = torch.arange(frames, dtype=torch.float32)
    return {
        "dataset": "test",
        "name": name,
        "root_motion": root,
        "body_motion": body,
        "body_feature_valid_mask": torch.ones_like(body, dtype=torch.bool),
        "text_data": [],
    }


class VariableLengthDataset(Dataset):
    def __init__(self):
        self.samples = [make_sample(20, "short"), make_sample(56, "valid")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_ldf_training_dataset_filters_samples_shorter_than_the_source_span():
    dataset = MinimumFrameDataset(VariableLengthDataset(), min_frames=40)
    assert len(dataset) == 1
    assert dataset.rejected_count == 1
    assert dataset[0]["name"] == "valid"


def test_parent_window_at_sequence_start_has_no_fake_encoder_context():
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=8,
        training=True,
    )([make_sample(40)])

    assert batch["root_motion"].shape == (1, 40, 5)
    assert batch["source_start_token"].tolist() == [0]
    assert batch["context_token_count"].tolist() == [0]
    assert not batch["previous_root_valid_mask"].item()
    assert batch["body_with_context"].shape == (1, 40, 265)
    assert torch.equal(batch["body_motion"][0, :, 0], torch.arange(40))


def test_continuation_carries_real_vae_context_without_translation_rebase(monkeypatch):
    monkeypatch.setattr(
        "utils.training.ldf.data.random.randint",
        lambda low, high: low if low == high else 2,
    )
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=2,
        training=True,
    )([make_sample(56)])

    assert batch["source_start_token"].tolist() == [2]
    assert batch["context_token_count"].tolist() == [2]
    assert batch["previous_root_valid_mask"].item()
    assert torch.equal(batch["body_with_context"][0, :8, 0], torch.arange(8))
    assert torch.equal(batch["body_motion"][0, :, 0], torch.arange(8, 48))
    assert torch.equal(
        batch["previous_root_frame"][0],
        torch.tensor([7.0, 0.0, 14.0, 1.0, 0.0]),
    )
    # Translation anchoring belongs to the rollout plan, not the collator.
    assert torch.equal(
        batch["root_motion"][0, 0],
        torch.tensor([8.0, 0.0, 16.0, 1.0, 0.0]),
    )


def test_each_sample_keeps_its_natural_parent_length_and_uses_right_padding():
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=48,
        generation_tokens=5,
        encoder_context_tokens=2,
        training=True,
    )([make_sample(40, "first"), make_sample(48, "second")])

    assert batch["span_token_count"].tolist() == [10, 12]
    assert batch["root_motion"].shape == (2, 48, 5)
    assert batch["frame_valid_mask"][0, :40].all()
    assert not batch["frame_valid_mask"][0, 40:].any()
    assert batch["frame_valid_mask"][1].all()
    assert batch["source_start_token"].tolist() == [0, 0]
    assert batch["context_token_count"].tolist() == [0, 0]
    assert batch["body_with_context"].shape == (2, 48, 265)
    assert batch["body_with_context_frame_valid_mask"][0, :40].all()
    assert not batch["body_with_context_frame_valid_mask"][0, 40:].any()
    assert batch["body_with_context_frame_valid_mask"][1].all()
    root_patches = batch["frame_valid_mask"].reshape(2, -1, 4)
    encoder_patches = batch["body_with_context_frame_valid_mask"].reshape(2, -1, 4)
    assert torch.equal(root_patches, root_patches[..., :1].expand_as(root_patches))
    assert torch.equal(
        encoder_patches,
        encoder_patches[..., :1].expand_as(encoder_patches),
    )
    root_tokens = root_patches[..., 0].sum(dim=1)
    encoder_active_tokens = (
        encoder_patches[..., 0].sum(dim=1) - batch["context_token_count"]
    )
    assert torch.equal(root_tokens, encoder_active_tokens)


def test_humanml_span_selects_one_caption_alternative_for_prompt_timeline(monkeypatch):
    monkeypatch.setattr(
        "utils.training.ldf.data.random.randint",
        lambda low, high: low if low == high else 1,
    )
    monkeypatch.setattr(
        "utils.training.ldf.data.random.choice", lambda alternatives: alternatives[-1]
    )
    sample = make_sample(48)
    sample["dataset"] = "HumanML3D"
    sample["text_data"] = [
        {"text": "before", "tokens": [], "start_frame": 0, "end_frame": 4},
        {"text": "overlap", "tokens": ["overlap"], "start_frame": 2, "end_frame": 10},
        {"text": "first", "tokens": ["first"], "start_frame": 4, "end_frame": 12},
        {"text": "alternative", "tokens": ["alt"], "start_frame": 4, "end_frame": 12},
        {"text": "tail", "tokens": ["tail"], "start_frame": 42, "end_frame": 48},
    ]
    prompt_timeline = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=0,
        training=True,
    )([sample])["prompt_timeline"][0]

    assert prompt_timeline == ["alternative"] * 10


def test_babel_span_compiles_one_prompt_per_motion_token(monkeypatch):
    monkeypatch.setattr(
        "utils.training.ldf.data.random.randint",
        lambda low, high: low if low == high else 1,
    )
    sample = make_sample(48)
    sample["dataset"] = "BABEL"
    sample["text_data"] = [
        {"text": "walk", "tokens": [], "start_frame": 0, "end_frame": 8},
        {"text": "turn", "tokens": [], "start_frame": 8, "end_frame": 48},
    ]
    prompt_timeline = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=0,
        training=True,
    )([sample])["prompt_timeline"][0]

    assert prompt_timeline == ["walk"] + ["turn"] * 9


def test_babel_arbitrary_frame_intervals_use_maximum_token_overlap():
    sample = make_sample(40)
    sample["dataset"] = "BABEL"
    sample["text_data"] = [
        {"text": "walk", "tokens": [], "start_frame": 0, "end_frame": 5},
        {"text": "turn", "tokens": [], "start_frame": 5, "end_frame": 40},
        # Equal overlap on token [4,8), but this shorter interval is more specific.
        {"text": "step", "tokens": [], "start_frame": 4, "end_frame": 8},
    ]
    timeline = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=0,
        training=False,
    )([sample])["prompt_timeline"][0]

    assert timeline == ["walk", "step"] + ["turn"] * 8


def test_length_bucket_sampler_does_not_mix_short_and_long_clips():
    class BucketDataset(Dataset):
        def __init__(self):
            self.samples = [
                make_sample(40, "40"),
                make_sample(44, "44"),
                make_sample(100, "100"),
                make_sample(104, "104"),
            ]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    dataset = MinimumFrameDataset(BucketDataset(), min_frames=40)
    sampler = LengthBucketBatchSampler(
        dataset,
        batch_size=2,
        bucket_width_frames=20,
        max_frames=200,
        seed=3,
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert all(
        max(dataset.frame_counts[index] for index, _, _ in batch)
        - min(dataset.frame_counts[index] for index, _, _ in batch)
        < 20
        for batch in batches
    )


def test_length_bucket_sampler_epoch_is_owned_by_set_epoch():
    dataset = MinimumFrameDataset(VariableLengthDataset(), min_frames=40)
    sampler = LengthBucketBatchSampler(
        dataset,
        batch_size=1,
        bucket_width_frames=20,
        max_frames=200,
        seed=3,
    )
    sampler.set_epoch(7)
    first = list(sampler)
    second = list(sampler)
    assert first == second
    assert sampler.epoch == 7


def test_length_bucket_sampler_builds_equal_nonoverlapping_ddp_steps():
    class TenSampleDataset(Dataset):
        def __init__(self):
            self.samples = [make_sample(40, str(index)) for index in range(10)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    dataset = MinimumFrameDataset(TenSampleDataset(), min_frames=40)
    samplers = [
        LengthBucketBatchSampler(
            dataset,
            batch_size=2,
            bucket_width_frames=20,
            max_frames=40,
            seed=7,
            rank=rank,
            num_replicas=2,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]
    assert [len(batches) for batches in rank_batches] == [3, 3]
    assert all(len(batch) == 2 for batches in rank_batches for batch in batches)
    all_indices = [
        index
        for batches in rank_batches
        for batch in batches
        for index, _, _ in batch
    ]
    # Ten real samples fill twelve distributed slots.  The final two slots are
    # deterministic padding, while every real sample remains represented.
    assert len(all_indices) == 12
    assert set(all_indices) == set(range(10))
    seeds_by_index = {}
    for batches in rank_batches:
        for batch in batches:
            for index, augmentation_seed, _ in batch:
                seeds_by_index.setdefault(index, set()).add(augmentation_seed)
    assert any(len(seeds) > 1 for seeds in seeds_by_index.values())
    for step in range(len(rank_batches[0])):
        assert len(
            {
                batch_seed
                for batches in rank_batches
                for _, _, batch_seed in batches[step]
            }
        ) == 1

    collator = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=0,
        training=True,
        cold_start_replay=0.5,
    )
    replay_by_rank = [
        [
            collator([dataset[index] for index in batch])["cold_start_replay"]
            for batch in batches
        ]
        for batches in rank_batches
    ]
    assert replay_by_rank[0] == replay_by_rank[1]


def test_distributed_validation_sampler_covers_each_sample_once():
    dataset = list(range(7))
    shards = [
        list(DistributedShardSampler(dataset, rank=rank, num_replicas=3))
        for rank in range(3)
    ]
    assert [len(shard) for shard in shards] == [3, 2, 2]
    assert sorted(index for shard in shards for index in shard) == list(range(7))


def test_ordered_seed_mixer_avoids_previous_weighted_sum_collision():
    # Both pairs produced 50 under sum((index + 1) * seed).
    assert _mix_seed([10, 20]) != _mix_seed([12, 19])
    assert _mix_seed([10, 20]) != _mix_seed([20, 10])


def test_seeded_training_batch_reproduces_crop_caption_and_yaw():
    sample = make_sample(56)
    sample["dataset"] = "HumanML3D"
    sample["_augmentation_seed"] = 91
    sample["text_data"] = [
        {"text": "first", "tokens": [], "start_frame": 0, "end_frame": 56},
        {"text": "second", "tokens": [], "start_frame": 0, "end_frame": 56},
    ]
    collator = LDFSpanCollator(
        min_frames=40,
        max_frames=48,
        generation_tokens=5,
        encoder_context_tokens=2,
        training=True,
        random_yaw=True,
    )
    first = collator([sample])
    second = collator([sample])
    assert torch.equal(first["root_motion"], second["root_motion"])
    assert torch.equal(first["body_motion"], second["body_motion"])
    assert first["source_start_token"].tolist() == second["source_start_token"].tolist()
    assert first["prompt_timeline"] == second["prompt_timeline"]


def test_cold_start_replay_forces_one_global_batch_to_the_true_sequence_start():
    samples = [make_sample(80, f"sample-{index}") for index in range(2)]
    for index, sample in enumerate(samples):
        sample["_augmentation_seed"] = 100 + index
        sample["_ldf_batch_seed"] = 77
    collator = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=4,
        training=True,
        cold_start_replay=1.0,
    )

    batch = collator(samples)

    assert batch["cold_start_replay"] is True
    assert batch["source_start_token"].tolist() == [0, 0]
    assert batch["context_token_count"].tolist() == [0, 0]
    assert not batch["previous_root_valid_mask"].any()


def test_cold_start_replay_rejects_rank_local_batch_seed_disagreement():
    samples = [make_sample(80, f"sample-{index}") for index in range(2)]
    for index, sample in enumerate(samples):
        sample["_augmentation_seed"] = 100 + index
        sample["_ldf_batch_seed"] = index
    collator = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=4,
        training=True,
        cold_start_replay=0.1,
    )

    with pytest.raises(ValueError, match="share a batch seed"):
        collator(samples)


def test_length_bucket_sampler_indices_flow_through_dataloader():
    dataset = MinimumFrameDataset(VariableLengthDataset(), min_frames=40)
    sampler = LengthBucketBatchSampler(
        dataset,
        batch_size=1,
        bucket_width_frames=20,
        max_frames=40,
        seed=5,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=LDFSpanCollator(
            min_frames=40,
            max_frames=40,
            generation_tokens=5,
            encoder_context_tokens=0,
            training=True,
        ),
    )
    batch = next(iter(loader))
    assert batch["name"] == ["valid"]
    assert batch["root_motion"].shape == (1, 40, 5)


def test_resumable_dataloader_restores_the_exact_next_batch():
    class FourSampleDataset(Dataset):
        def __init__(self):
            self.samples = [make_sample(40, f"sample-{index}") for index in range(4)]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    def create_loader():
        dataset = MinimumFrameDataset(FourSampleDataset(), min_frames=40)
        sampler = LengthBucketBatchSampler(
            dataset,
            batch_size=1,
            bucket_width_frames=20,
            max_frames=40,
            seed=5,
        )
        sampler.set_epoch(3)
        return ResumableDataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda samples: samples[0]["name"],
            num_workers=0,
        )

    original = create_loader()
    iterator = iter(original)
    consumed = [next(iterator), next(iterator)]
    state = original.state_dict()
    assert CombinedLoader(original, "max_size_cycle")._state_dicts() == [state]
    expected_remaining = list(iterator)

    resumed = create_loader()
    resumed.load_state_dict(state)
    assert list(resumed) == expected_remaining
    assert len(consumed) + len(expected_remaining) == 4

    resumed.batch_sampler.set_epoch(4)
    assert len(list(resumed)) == 4


def test_validation_parent_window_is_deterministic_for_all_probe_types():
    sample = make_sample(56)
    sample["dataset"] = "HumanML3D"
    sample["text_data"] = [
        {"text": "first", "tokens": [], "start_frame": 0, "end_frame": 56},
        {"text": "second", "tokens": [], "start_frame": 0, "end_frame": 56},
    ]
    cold_parent = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=2,
        training=False,
        validation_probe="teacher_cold",
    )([sample])
    continuation_collator = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        generation_tokens=5,
        encoder_context_tokens=2,
        training=False,
    )
    first = continuation_collator([sample])
    second = continuation_collator([sample])

    assert cold_parent["source_start_token"].tolist() == [0]
    assert cold_parent["context_token_count"].tolist() == [0]
    assert not cold_parent["previous_root_valid_mask"].item()
    assert first["source_start_token"].tolist() == [2]
    assert first["context_token_count"].tolist() == [2]
    assert torch.equal(first["root_motion"], second["root_motion"])
    assert first["prompt_timeline"][0] == ["first"] * 10


def test_continuation_validation_labels_early_middle_and_late_history_probes():
    samples = [make_sample(80, f"sample-{index}") for index in range(3)]
    for index, sample in enumerate(samples):
        sample["_ldf_sample_index"] = index
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=80,
        generation_tokens=5,
        encoder_context_tokens=2,
        training=False,
        validation_positions=("early", "middle", "late"),
    )(samples)

    assert batch["validation_position"] == ["early", "middle", "late"]
    assert batch["source_start_token"].tolist() == [0, 0, 0]
    assert batch["context_token_count"].tolist() == [0, 0, 0]


def test_create_dataloaders_exposes_named_validation_probes(monkeypatch):
    dataset = MinimumFrameDataset(VariableLengthDataset(), min_frames=20)
    monkeypatch.setattr(
        "utils.training.ldf.data.create_dataset", lambda _cfg, _split: dataset
    )
    cfg = OmegaConf.create(
        {
            "seed": 3,
            "train": True,
            "self_forcing": {
                "k_schedule": [[0, 1], [5, 2], [14, 5]],
                "teacher_replay": {2: 0.2, 5: 0.1},
                "cold_start_replay": 0.0,
                "cold_start": {
                    "persistent_probability": 0.5,
                    "rollout_commits": 2,
                },
            },
            "trainer": {"max_steps": 20},
            "training": {
                "window": {
                    "max_tokens": 10,
                    "generation_tokens": 5,
                }
            },
            "model": {"params": {"chunk_size": 5}},
            "data": {
                "min_frames": 20,
                "max_frames": 40,
                "random_yaw": False,
                "train_batch_size": 1,
                "val_batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            },
        }
    )
    train, validation = create_dataloaders(cfg, encoder_context_tokens=2)
    assert train is not None
    # K=5 reserves four rollout tokens, so the five-token short clip is
    # filtered from self-forcing training while remaining valid for K=1 cold val.
    assert next(iter(train))["name"] == ["valid"]
    validation_batches = [next(iter(loader)) for loader in validation]
    assert [batch["validation_probe"] for batch in validation_batches] == [
        "teacher_cold",
        "persistent_cold",
        "teacher_continuation",
        "self_forcing",
    ]
    assert validation_batches[0]["source_start_token"].tolist() == [0]
    assert validation_batches[0]["context_token_count"].tolist() == [0]
    assert not validation_batches[0]["previous_root_valid_mask"].any()
    assert validation_batches[1]["source_start_token"].tolist() == [0]
    assert validation_batches[1]["context_token_count"].tolist() == [0]
    assert not validation_batches[1]["previous_root_valid_mask"].any()
