import torch
from torch.utils.data import Dataset

from utils.training.ldf.data import LDFSpanCollator, MinimumFrameDataset


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
        self.samples = [make_sample(20, "short"), make_sample(40, "valid")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_ldf_training_dataset_filters_samples_shorter_than_the_source_span():
    dataset = MinimumFrameDataset(VariableLengthDataset(), min_frames=40)
    assert len(dataset) == 1
    assert dataset.rejected_count == 1
    assert dataset[0]["name"] == "valid"


def test_true_cold_start_is_the_real_sequence_start_without_hidden_boundary():
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        encoder_context_tokens=8,
        training=True,
        cold_start=True,
    )([make_sample(48)])

    assert batch["root_motion"].shape == (1, 40, 5)
    assert batch["source_start_token"].tolist() == [0]
    assert batch["cold_start_mask"].tolist() == [True]
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
        encoder_context_tokens=2,
        training=True,
        cold_start=False,
    )([make_sample(56)])

    assert batch["source_start_token"].tolist() == [2]
    assert batch["cold_start_mask"].tolist() == [False]
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


def test_span_length_is_batch_shared_while_source_crops_are_independent(monkeypatch):
    choices = iter((10, 1, 3))
    monkeypatch.setattr(
        "utils.training.ldf.data.random.randint", lambda *_: next(choices)
    )
    batch = LDFSpanCollator(
        min_frames=40,
        max_frames=48,
        encoder_context_tokens=2,
        training=True,
        cold_start=False,
    )([make_sample(56, "first"), make_sample(64, "second")])

    assert batch["span_token_count"] == 10
    assert batch["root_motion"].shape == (2, 40, 5)
    assert batch["frame_valid_mask"].all()
    assert batch["source_start_token"].tolist() == [1, 3]
    assert batch["context_token_count"].tolist() == [1, 2]
    assert batch["body_with_context"].shape == (2, 48, 265)
    assert batch["body_with_context_frame_valid_mask"][0, :44].all()
    assert not batch["body_with_context_frame_valid_mask"][0, 44:].any()
    assert batch["body_with_context_frame_valid_mask"][1].all()


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
        encoder_context_tokens=0,
        training=True,
        cold_start=False,
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
        encoder_context_tokens=0,
        training=True,
        cold_start=False,
    )([sample])["prompt_timeline"][0]

    assert prompt_timeline == ["walk"] + ["turn"] * 9


def test_validation_probes_are_deterministic_for_cold_and_continuation():
    sample = make_sample(56)
    sample["dataset"] = "HumanML3D"
    sample["text_data"] = [
        {"text": "first", "tokens": [], "start_frame": 0, "end_frame": 56},
        {"text": "second", "tokens": [], "start_frame": 0, "end_frame": 56},
    ]
    cold = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        encoder_context_tokens=2,
        training=False,
        cold_start=True,
    )([sample])
    continuation_collator = LDFSpanCollator(
        min_frames=40,
        max_frames=40,
        encoder_context_tokens=2,
        training=False,
        cold_start=False,
    )
    first = continuation_collator([sample])
    second = continuation_collator([sample])

    assert cold["source_start_token"].tolist() == [0]
    assert first["source_start_token"].tolist() == [2]
    assert torch.equal(first["root_motion"], second["root_motion"])
    assert first["prompt_timeline"][0] == ["first"] * 10
