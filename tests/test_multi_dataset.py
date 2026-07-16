import numpy as np
import torch

from datasets.babel import BABELDataset
from datasets.humanml3d import HumanML3DDataset
from datasets.multi import MultiDataset
from tools.compute_vae_stats import compute_motion_statistics
from tools.preprocess_babel import build_dataset
from tests.vae_helpers import make_vae
from utils.training.vae.data import VAEWindowCollator
from utils.training.vae.evaluation import evaluate_dataset


def make_motion(frames: int) -> np.ndarray:
    motion = np.zeros((frames, 263), dtype=np.float32)
    motion[:, 1] = 0.05
    motion[:, 3] = 1.0
    motion[:, 67:193] = np.tile(
        np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), 21
    )
    return motion


def write_npz(root, name):
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    root_motion = np.zeros((20, 5), dtype=np.float32)
    root_motion[:, 3] = 1
    body = np.zeros((20, 265), dtype=np.float32)
    np.savez_compressed(
        root / "artifacts" / f"{name}.npz",
        root_motion=root_motion,
        body_motion=body,
        body_feature_valid_mask=np.ones_like(body, dtype=bool),
        obsolete_metadata="ignored",
    )
    (root / "train.txt").write_text(f"{name}\n")
    texts = root / "texts"
    texts.mkdir()
    return texts


def test_babel_is_independent_and_parses_segmented_text(tmp_path):
    assert not issubclass(BABELDataset, HumanML3DDataset)
    texts = write_npz(tmp_path, "sample")
    (texts / "sample.txt").write_text(
        "walk#walk/VERB#0.0#0.4\nwalk ahead#walk/VERB ahead/ADV#0.0#0.4\n"
        "sit#sit/VERB#0.4#1.0\n"
    )
    dataset = BABELDataset(
        meta_paths=[tmp_path / "train.txt"], split="train", text_path=texts
    )
    assert set(dataset.dataset[0]) == {
        "dataset", "name", "motion_path", "text_path"
    }
    sample = dataset[0]
    assert sample["dataset"] == "BABEL"
    assert sample["root_motion"].shape == (20, 5)
    assert sample["body_motion"].shape == (20, 265)
    assert [(item["start_frame"], item["end_frame"]) for item in sample["text_data"]] == [
        (0, 8), (0, 8), (8, 20)
    ]


def test_babel_ignores_annotations_without_positive_motion_overlap(tmp_path):
    texts = write_npz(tmp_path, "sample")
    (texts / "sample.txt").write_text(
        "walk#walk/VERB#0.0#0.4\n"
        "zero#zero/NOUN#0.5#0.5\n"
        "reverse#reverse/VERB#1.0#0.5\n"
        "past#past/NOUN#2.0#3.0\n"
    )
    dataset = BABELDataset(
        meta_paths=[tmp_path / "train.txt"], split="train", text_path="texts"
    )

    assert [item["text"] for item in dataset[0]["text_data"]] == ["walk"]


def test_babel_preprocess_copies_text_and_writes_minimal_motion(tmp_path):
    source = tmp_path / "BABEL_streamed"
    (source / "motions").mkdir(parents=True)
    (source / "texts").mkdir()
    np.save(source / "motions" / "sample.npy", make_motion(20))
    (source / "texts" / "sample.txt").write_text("walk#walk/VERB#0#1\n")
    (source / "train_processed.txt").write_text("sample\n")
    (source / "val_processed.txt").write_text("sample\n")
    output = tmp_path / "BABEL_motion"
    summary = build_dataset(source, output, workers=1)
    assert summary["copied_texts"] == 1
    assert (output / "all.txt").read_text() == "sample\n"
    assert (output / "texts" / "sample.txt").read_text() == (
        "walk#walk/VERB#0#1\n"
    )
    with np.load(output / "artifacts" / "sample.npz") as data:
        assert set(data.files) == {
            "root_motion", "body_motion", "body_feature_valid_mask"
        }


def test_multi_dataset_only_concatenates_and_preserves_source_identity(tmp_path):
    human_root = tmp_path / "HumanML3D_motion"
    babel_root = tmp_path / "BABEL_motion"
    human_text = write_npz(human_root, "human")
    babel_text = write_npz(babel_root, "babel")
    (human_text / "human.txt").write_text("human#human/NOUN#0#0\n")
    (babel_text / "babel.txt").write_text("babel#babel/NOUN#0#1\n")
    dataset = MultiDataset(
        dataset_configs=[
            {
                "target": "datasets.humanml3d.HumanML3DDataset",
                "train_meta_paths": [human_root / "train.txt"],
                "text_path": human_text,
            },
            {
                "target": "datasets.babel.BABELDataset",
                "train_meta_paths": [babel_root / "train.txt"],
                "text_path": babel_text,
            },
        ],
        split="train",
    )
    assert dataset.dataset_lengths == (1, 1)
    assert dataset.split == "train"
    assert [dataset[index]["dataset"] for index in range(2)] == [
        "HumanML3D", "BABEL"
    ]
    batch = VAEWindowCollator(
        min_frames=20, max_frames=20, training=False
    )([dataset[0], dataset[1]])
    assert batch["body_motion"].shape == (2, 20, 265)
    assert batch["dataset"] == ["HumanML3D", "BABEL"]
    statistics = compute_motion_statistics(dataset, random_yaw=False)
    assert statistics["body_cont_mean"].shape == (261,)

    model = make_vae(
        latent_dim=4,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        with_latent_stats=False,
    ).eval()
    single_summary = evaluate_dataset(
        model,
        dataset_name="humanml3d",
        dataset=dataset.datasets[0],
        sample_count=1,
        output_root=tmp_path / "eval_single",
        model_id="tiny_vae",
        device=torch.device("cpu"),
        parity_atol=1e-5,
        render_video=False,
        render_fps=20,
        mode="stream",
    )
    multi_summary = evaluate_dataset(
        model,
        dataset_name="multi",
        dataset=dataset,
        sample_count=2,
        output_root=tmp_path / "eval_multi",
        model_id="tiny_vae",
        device=torch.device("cpu"),
        parity_atol=1e-5,
        render_video=False,
        render_fps=20,
        mode="stream",
    )
    assert single_summary["sample_count"] == 1
    assert multi_summary["sample_count"] == 2


def test_babel_source_identity_does_not_depend_on_directory_name(tmp_path):
    renamed_root = tmp_path / "another_release_name"
    texts = write_npz(renamed_root, "sample")
    (texts / "sample.txt").write_text("walk#walk/VERB#0#1\n")
    dataset = BABELDataset(
        meta_paths=[renamed_root / "train.txt"],
        split="train",
        text_path="texts",
    )
    assert dataset[0]["dataset"] == "BABEL"
