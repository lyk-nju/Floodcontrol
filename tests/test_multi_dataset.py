import numpy as np

from datasets.babel import BABELDataset
from datasets.multi import MultiDataset, collate_multi
from tools.motion_artifact import process_file
from tools.preprocess_babel import build_dataset


def make_motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, 263), dtype=np.float32)
    motion[:, 1] = 0.05
    motion[:, 3] = 1.0
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 67:193] = np.tile(identity_6d, 21)
    return motion


def test_babel_motion_builder_is_resumable_and_filters_short_clips(tmp_path):
    source = tmp_path / "BABEL_streamed"
    motions = source / "motions"
    motions.mkdir(parents=True)
    np.save(motions / "sample.npy", make_motion(20))
    np.save(motions / "short.npy", make_motion(8))
    (source / "train_processed.txt").write_text("sample\nshort\n")
    (source / "val_processed.txt").write_text("sample\n")
    output = tmp_path / "BABEL_motion"

    summary = build_dataset(source, output, workers=1)
    assert summary["source_dataset"] == "BABEL_streamed"
    assert summary["splits"] == {"train": 1, "val": 1}
    assert summary["too_short"] == {"train": 1, "val": 0}
    assert summary["converted"] == 1
    assert (output / "train.txt").read_text() == "sample\n"
    assert (output / "val.txt").read_text() == "sample\n"
    assert not (output / "test.txt").exists()

    resumed = build_dataset(source, output, workers=1)
    assert resumed["converted"] == 0
    assert resumed["skipped"] == 1
    dataset = BABELDataset(
        meta_paths=[output / "train.txt"],
        split="train",
        min_frames=20,
        max_frames=20,
    )
    assert dataset[0]["body_motion"].shape == (20, 265)
    assert dataset[0]["dataset"] == "BABEL_motion"


def test_multi_dataset_composes_only_the_shared_vae_contract(tmp_path):
    human_root = tmp_path / "HumanML3D_motion"
    babel_root = tmp_path / "BABEL_motion"
    for root, sample in ((human_root, "human"), (babel_root, "babel")):
        (root / "artifacts").mkdir(parents=True)
        source = tmp_path / f"{sample}.npy"
        np.save(source, make_motion(20))
        process_file(source, root / "artifacts" / f"{sample}.npz")
        (root / "train.txt").write_text(f"{sample}\n")

    dataset = MultiDataset(
        dataset_configs=[
            {
                "target": "datasets.humanml3d.HumanML3DDataset",
                "train_meta_paths": [human_root / "train.txt"],
            },
            {
                "target": "datasets.babel.BABELDataset",
                "train_meta_paths": [babel_root / "train.txt"],
            },
        ],
        split="train",
        min_frames=20,
        max_frames=20,
    )
    assert dataset.dataset_lengths == (1, 1)
    batch = collate_multi([dataset[0], dataset[1]])
    assert batch["body_motion"].shape == (2, 20, 265)
    assert batch["root_motion"].shape == (2, 20, 5)
    assert batch["frame_valid_mask"].all()
    assert batch["dataset"] == ["HumanML3D_motion", "BABEL_motion"]
    assert set(item["dataset"] for item in (dataset[0], dataset[1])) == {
        "HumanML3D_motion",
        "BABEL_motion",
    }
