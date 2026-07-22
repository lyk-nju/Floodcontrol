"""Measure Root5/Body259 round-trip, global-yaw invariance, and evaluator drift."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from metrics.humanml import convert_root5_body259_to_humanml263
from metrics.tools.t2m_evaluator import MovementConvEncoder, MotionEncoderBiGRUCo
from metrics.tools.utils import (
    calculate_activation_statistics_np,
    calculate_frechet_distance_np,
)
from tools.convert_motion_263_to_259 import convert_motion_263_to_259
from utils.motion_process import rotate_motion_yaw


FEATURE_BLOCKS = {
    "root": slice(0, 4),
    "positions": slice(4, 67),
    "rotations": slice(67, 193),
    "velocities": slice(193, 259),
    "contacts": slice(259, 263),
}


class ErrorAccumulator:
    def __init__(self):
        self.absolute_sum = {name: 0.0 for name in FEATURE_BLOCKS}
        self.count = {name: 0 for name in FEATURE_BLOCKS}
        self.maximum = {name: 0.0 for name in FEATURE_BLOCKS}

    def update(self, reference: torch.Tensor, converted: torch.Tensor) -> None:
        for name, feature_slice in FEATURE_BLOCKS.items():
            error = (reference[..., feature_slice] - converted[..., feature_slice]).abs()
            self.absolute_sum[name] += float(error.double().sum())
            self.count[name] += error.numel()
            self.maximum[name] = max(self.maximum[name], float(error.max()))

    def result(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "mae": self.absolute_sum[name] / self.count[name],
                "max_abs": self.maximum[name],
            }
            for name in FEATURE_BLOCKS
        }


class T2MEmbedder:
    def __init__(self, deps: Path, device: torch.device):
        self.device = device
        self.mean = torch.from_numpy(np.load(deps / "t2m/meta/mean.npy")).float().to(device)
        self.std = torch.from_numpy(np.load(deps / "t2m/meta/std.npy")).float().to(device)
        self.movement = MovementConvEncoder(259, 512, 512).to(device).eval()
        self.motion = MotionEncoderBiGRUCo(512, 1024, 512).to(device).eval()
        self.movement.load_state_dict(
            torch.load(
                deps / "t2m/humanml3d/movement_encoder.pt",
                map_location=device,
                weights_only=True,
            )
        )
        self.motion.load_state_dict(
            torch.load(
                deps / "t2m/humanml3d/motion_encoder.pt",
                map_location=device,
                weights_only=True,
            )
        )

    @torch.no_grad()
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        squeeze = features.ndim == 2
        if squeeze:
            features = features[None]
        if features.ndim != 3 or features.shape[-1] != 263:
            raise ValueError("features must be [F,263] or [B,F,263]")
        features = features.to(self.device)
        normalized = (features - self.mean) / self.std
        movement = self.movement(normalized[..., :-4])
        movement_length = features.shape[1] // 4
        embedding = self.motion(
            movement,
            torch.full(
                (features.shape[0],),
                movement_length,
                device=self.device,
                dtype=torch.long,
            ),
        )
        embedding = embedding.cpu()
        return embedding[0] if squeeze else embedding


def _fid(reference: list[torch.Tensor], converted: list[torch.Tensor]) -> float:
    first = torch.stack(reference).numpy()
    second = torch.stack(converted).numpy()
    first_mean, first_cov = calculate_activation_statistics_np(first)
    second_mean, second_cov = calculate_activation_statistics_np(second)
    # Numerical sqrtm noise can produce a tiny negative value for identical
    # distributions even though the mathematical Fréchet distance is >= 0.
    return max(
        0.0,
        float(calculate_frechet_distance_np(first_mean, first_cov, second_mean, second_cov)),
    )


def _embedding_metrics(
    reference: list[torch.Tensor],
    converted: list[torch.Tensor],
) -> dict[str, float]:
    first = torch.stack(reference)
    second = torch.stack(converted)
    return {
        "embedding_l2_mean": float((first - second).norm(dim=-1).mean()),
        "embedding_cosine_mean": float(
            torch.nn.functional.cosine_similarity(first, second).mean()
        ),
        "fid": _fid(reference, converted),
    }


def _yaw_key(degrees: float) -> str:
    value = float(degrees)
    if value.is_integer():
        return f"yaw_{int(value):+04d}_degrees"
    return f"yaw_{value:+08.3f}_degrees".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--motion-root", required=True)
    parser.add_argument("--deps", required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    parser.add_argument(
        "--yaw-degrees",
        type=float,
        nargs="*",
        default=(0.0, 45.0, 90.0, 180.0),
        help=(
            "Global yaw offsets evaluated after 263 -> Root5/Body259 conversion. "
            "Each rotated result is converted back to canonical HumanML263."
        ),
    )
    parser.add_argument(
        "--random-yaw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also evaluate one deterministic uniform yaw per sample.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.samples <= 1:
        raise ValueError("samples must be greater than one")
    if not args.yaw_degrees and not args.random_yaw:
        raise ValueError("enable at least one fixed or random yaw evaluation")
    if any(not math.isfinite(value) for value in args.yaw_degrees):
        raise ValueError("--yaw-degrees must contain only finite values")
    if len(set(float(value) for value in args.yaw_degrees)) != len(args.yaw_degrees):
        raise ValueError("--yaw-degrees must not contain duplicates")

    names = [line.strip() for line in Path(args.split).read_text().splitlines() if line.strip()]
    embedder = T2MEmbedder(Path(args.deps), torch.device(args.device))
    exact_error = ErrorAccumulator()
    approximate_error = ErrorAccumulator()
    exact_reference_embeddings = []
    exact_converted_embeddings = []
    approximate_reference_embeddings = []
    approximate_converted_embeddings = []
    yaw_errors = {
        _yaw_key(degrees): ErrorAccumulator() for degrees in args.yaw_degrees
    }
    yaw_embeddings: dict[str, list[torch.Tensor]] = {
        key: [] for key in yaw_errors
    }
    yaw_angles: dict[str, list[float]] = {
        key: [float(degrees)]
        for key, degrees in zip(yaw_errors, args.yaw_degrees, strict=True)
    }
    if args.random_yaw:
        yaw_errors["yaw_uniform_random"] = ErrorAccumulator()
        yaw_embeddings["yaw_uniform_random"] = []
        yaw_angles["yaw_uniform_random"] = []
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    processed = 0
    for name in names:
        path = Path(args.motion_root) / f"{name}.npy"
        if not path.is_file():
            continue
        source = torch.from_numpy(np.load(path)).float()
        if source.ndim != 2 or source.shape[-1] != 263 or source.shape[0] < 8:
            continue
        root, body, _ = convert_motion_263_to_259(source)
        exact = convert_root5_body259_to_humanml263(root, body, tail="drop")
        approximate = convert_root5_body259_to_humanml263(
            root, body, tail="approximate"
        )
        exact_reference = source[:-1]
        exact_error.update(exact_reference, exact)
        approximate_error.update(source, approximate)
        yaw_values = {
            _yaw_key(degrees): math.radians(float(degrees))
            for degrees in args.yaw_degrees
        }
        if args.random_yaw:
            yaw = float(torch.rand((), generator=generator) * (2.0 * torch.pi))
            yaw_values["yaw_uniform_random"] = yaw
            yaw_angles["yaw_uniform_random"].append(math.degrees(yaw))
        rotated_features: dict[str, torch.Tensor] = {}
        for key, yaw in yaw_values.items():
            rotated_root, rotated_body = rotate_motion_yaw(
                root[None], body[None], root.new_tensor([yaw])
            )
            rotated = convert_root5_body259_to_humanml263(
                rotated_root[0], rotated_body[0], tail="drop"
            )
            yaw_errors[key].update(exact, rotated)
            rotated_features[key] = rotated

        exact_group = [exact_reference, exact, *rotated_features.values()]
        exact_group_embeddings = embedder(torch.stack(exact_group))
        exact_reference_embeddings.append(exact_group_embeddings[0])
        exact_converted_embeddings.append(exact_group_embeddings[1])
        for offset, key in enumerate(rotated_features, start=2):
            yaw_embeddings[key].append(exact_group_embeddings[offset])

        approximate_group_embeddings = embedder(torch.stack([source, approximate]))
        approximate_reference_embeddings.append(approximate_group_embeddings[0])
        approximate_converted_embeddings.append(approximate_group_embeddings[1])
        processed += 1
        if processed % 100 == 0 or processed == args.samples:
            print(
                f"processed {processed}/{min(args.samples, len(names))} motions",
                flush=True,
            )
        if processed >= args.samples:
            break
    if processed < 2:
        raise RuntimeError("not enough valid HumanML motions were found")

    global_yaw = {}
    for key in yaw_errors:
        angles = yaw_angles[key]
        global_yaw[key] = {
            "angle_degrees": (
                angles[0]
                if key != "yaw_uniform_random"
                else {
                    "sampling": "uniform_[0,360)",
                    "seed": int(args.seed),
                    "minimum": min(angles),
                    "maximum": max(angles),
                    "mean": sum(angles) / len(angles),
                }
            ),
            "versus_unrotated_roundtrip": {
                "features": yaw_errors[key].result(),
                **_embedding_metrics(
                    exact_converted_embeddings, yaw_embeddings[key]
                ),
            },
            "versus_source": _embedding_metrics(
                exact_reference_embeddings, yaw_embeddings[key]
            ),
        }

    result = {
        "samples": processed,
        "exact_drop": {
            "features": exact_error.result(),
            **_embedding_metrics(
                exact_reference_embeddings, exact_converted_embeddings
            ),
        },
        "approximate_tail": {
            "features": approximate_error.result(),
            **_embedding_metrics(
                approximate_reference_embeddings,
                approximate_converted_embeddings,
            ),
        },
        "drop_against_full_length": {
            **_embedding_metrics(
                approximate_reference_embeddings, exact_converted_embeddings
            ),
        },
        "global_yaw": global_yaw,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
