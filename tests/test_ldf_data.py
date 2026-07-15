import torch

from utils.training.ldf.data import LDFWindowCollator


def make_sample(frames=16, name="sample"):
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


def test_ldf_validation_has_fixed_left_padding_and_deterministic_prefix():
    sample = make_sample(12)
    batch = LDFWindowCollator(
        min_frames=8,
        max_frames=8,
        encoder_context_tokens=2,
        training=False,
    )([sample])
    assert batch["root_motion"].shape == (1, 8, 5)
    assert batch["body_with_context"].shape == (1, 16, 265)
    assert not batch["context_frame_valid_mask"].any()
    assert not batch["body_with_context_frame_valid_mask"][0, :8].any()
    assert batch["body_with_context_frame_valid_mask"][0, 8:].all()
    assert torch.equal(batch["body_with_context"][0, :8], torch.zeros(8, 265))
    assert torch.equal(batch["body_with_context"][0, 8:, 0], torch.arange(8))
    assert not batch["previous_root_valid_mask"].item()
    assert batch["context_token_count"] == 2


def test_ldf_training_carries_complete_context_and_previous_root(monkeypatch):
    choices = iter((2, 2))  # active 8 frames, starting at frame 8
    monkeypatch.setattr("utils.training.ldf.data.random.randint", lambda *_: next(choices))
    sample = make_sample(20)
    batch = LDFWindowCollator(
        min_frames=8,
        max_frames=8,
        encoder_context_tokens=2,
        training=True,
    )([sample])
    assert batch["context_frame_valid_mask"].all()
    assert torch.equal(batch["body_with_context"][0, :8, 0], torch.arange(8))
    assert torch.equal(batch["body_motion"][0, :, 0], torch.arange(8, 16))
    assert batch["previous_root_valid_mask"].item()
    # Active frame 8 is the translation origin, so the preceding frame is
    # one x step and two z steps behind it.
    assert torch.equal(
        batch["previous_root_frame"][0],
        torch.tensor([-1.0, 0.0, -2.0, 1.0, 0.0]),
    )
    assert torch.equal(
        batch["root_motion"][0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
    )


def test_ldf_batch_padding_keeps_active_and_encoder_masks_separate():
    long = make_sample(12, "long")
    short = make_sample(8, "short")
    batch = LDFWindowCollator(
        min_frames=8,
        max_frames=12,
        encoder_context_tokens=1,
        training=False,
    )([long, short])
    assert batch["body_motion"].shape == (2, 12, 265)
    assert batch["body_with_context"].shape == (2, 16, 265)
    assert batch["frame_valid_mask"][0].all()
    assert batch["frame_valid_mask"][1, :8].all()
    assert not batch["frame_valid_mask"][1, 8:].any()
    assert not batch["body_with_context_frame_valid_mask"][1, 12:].any()
    assert not batch["body_with_context_feature_valid_mask"][1, 12:].any()


def test_ldf_text_intersection_token_clipping_and_caption_alternatives(monkeypatch):
    choices = iter((2, 1))  # active [4,12)
    monkeypatch.setattr("utils.training.ldf.data.random.randint", lambda *_: next(choices))
    monkeypatch.setattr(
        "utils.training.ldf.data.random.choice", lambda alternatives: alternatives[-1]
    )
    sample = make_sample(16)
    sample["text_data"] = [
        {"text": "before", "tokens": [], "start_frame": 0, "end_frame": 4},
        {"text": "overlap", "tokens": ["overlap"], "start_frame": 2, "end_frame": 10},
        {"text": "first", "tokens": ["first"], "start_frame": 4, "end_frame": 12},
        {"text": "alternative", "tokens": ["alt"], "start_frame": 4, "end_frame": 12},
        {"text": "tail", "tokens": ["tail"], "start_frame": 10, "end_frame": 16},
    ]
    text_data = LDFWindowCollator(
        min_frames=8,
        max_frames=8,
        encoder_context_tokens=0,
        training=True,
    )([sample])["text_data"][0]
    assert [item["text"] for item in text_data] == [
        "overlap", "alternative", "tail"
    ]
    assert [(item["start_token"], item["end_token"]) for item in text_data] == [
        (0, 2), (0, 2), (1, 2)
    ]


def test_ldf_validation_chooses_first_caption_for_same_interval():
    sample = make_sample(8)
    sample["text_data"] = [
        {"text": "first", "tokens": [], "start_frame": 0, "end_frame": 8},
        {"text": "second", "tokens": [], "start_frame": 0, "end_frame": 8},
    ]
    collator = LDFWindowCollator(
        min_frames=8,
        max_frames=8,
        encoder_context_tokens=0,
        training=False,
    )
    assert collator([sample])["text_data"][0][0]["text"] == "first"
    assert collator([sample])["text_data"][0][0]["text"] == "first"
