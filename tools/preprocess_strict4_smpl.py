"""Build strict4 root5/body265 artifacts from native 22-joint rotations.

Input NPZ files must contain ``local_rotations`` [F,22,3,3],
``root_translation`` [F,3], ``parents`` [22], ``offsets`` [22,3], and ``fps``.
This tool deliberately has no position-only IK or HumanML263 fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.conditions.vae import CONTRACT_VERSION, FRAMES_PER_TOKEN, NUM_JOINTS
from utils.motion_representation import build_root_body_motion, matrix_to_rotation_6d, rotation_6d_to_matrix


def forward_kinematics(
    local_rotations: torch.Tensor,
    root_translation: torch.Tensor,
    parents: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(local_rotations.shape[-3:]) != (NUM_JOINTS, 3, 3):
        raise ValueError("native local_rotations must be [F,22,3,3]")
    if parents.tolist()[0] not in (-1, 0):
        raise ValueError("joint zero must be the skeleton root")
    global_rotations = []
    global_positions = []
    for joint in range(NUM_JOINTS):
        parent = int(parents[joint])
        if joint == 0:
            rotation = local_rotations[:, joint]
            position = root_translation
        else:
            if parent < 0 or parent >= joint:
                raise ValueError("parents must be topologically ordered")
            rotation = global_rotations[parent] @ local_rotations[:, joint]
            position = global_positions[parent] + torch.einsum(
                "fij,j->fi", global_rotations[parent], offsets[joint]
            )
        global_rotations.append(rotation)
        global_positions.append(position)
    return torch.stack(global_rotations, dim=1), torch.stack(global_positions, dim=1)


def resample_native(
    rotations: torch.Tensor,
    translation: torch.Tensor,
    source_fps: float,
    target_fps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if abs(float(source_fps) - float(target_fps)) < 1e-6:
        return rotations, translation
    duration = (rotations.shape[0] - 1) / float(source_fps)
    target_frames = int(round(duration * float(target_fps))) + 1
    source_position = torch.linspace(0, rotations.shape[0] - 1, target_frames)
    left = source_position.floor().long()
    right = source_position.ceil().long().clamp(max=rotations.shape[0] - 1)
    weight = (source_position - left).to(rotations.dtype)
    translation = torch.lerp(translation[left], translation[right], weight[:, None])
    # Interpolate the native orientation representation, then project back to SO(3).
    # This preserves source rotations and never derives them from joint positions.
    six = matrix_to_rotation_6d(rotations)
    six = torch.lerp(six[left], six[right], weight[:, None, None])
    return rotation_6d_to_matrix(six), translation


def process_file(source: Path, target: Path, *, target_fps: float) -> dict:
    with np.load(source, allow_pickle=False) as data:
        required = {"local_rotations", "root_translation", "parents", "offsets", "fps"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(
                f"STRICT4_NATIVE_ROTATIONS_REQUIRED: {source} misses {missing}; "
                "position-only and legacy263 inputs are unsupported"
            )
        rotations = torch.from_numpy(data["local_rotations"]).float()
        translation = torch.from_numpy(data["root_translation"]).float()
        parents = torch.from_numpy(data["parents"]).long()
        offsets = torch.from_numpy(data["offsets"]).float()
        source_fps = float(np.asarray(data["fps"]).item())
    if rotations.shape[1] != NUM_JOINTS or tuple(parents.shape) != (NUM_JOINTS,) or tuple(offsets.shape) != (NUM_JOINTS, 3):
        raise ValueError("strict4 preprocessing requires one retargeted 22-joint skeleton")
    rotations, translation = resample_native(rotations, translation, source_fps, target_fps)
    usable = rotations.shape[0] // FRAMES_PER_TOKEN * FRAMES_PER_TOKEN
    if usable < FRAMES_PER_TOKEN:
        raise ValueError(f"{source} has fewer than four usable frames")
    rotations, translation = rotations[:usable], translation[:usable]
    global_rotations, global_positions = forward_kinematics(
        rotations, translation, parents, offsets
    )
    heading = torch.atan2(global_rotations[:, 0, 0, 2], global_rotations[:, 0, 0, 0])
    root, body, feature_valid = build_root_body_motion(
        global_positions[None], global_rotations[None], translation[None],
        heading[None], fps=target_fps
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        root_motion=root[0].numpy(),
        body_motion=body[0].numpy(),
        body_feature_valid_mask=feature_valid[0].numpy(),
        contract_version=CONTRACT_VERSION,
        fps=np.float32(target_fps),
    )
    return {"frames": usable, "source_fps": source_fps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="folder of native-rotation NPZ files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--target-fps", type=float, default=20.0)
    args = parser.parse_args()
    source_root = Path(args.source)
    if not source_root.is_dir():
        raise RuntimeError(
            "STRICT4_NATIVE_ROTATIONS_REQUIRED: --source must point to native SMPL/AMASS rotation NPZ files"
        )
    output = Path(args.output)
    artifact_root = output / "artifacts"
    records = []
    for source in sorted(source_root.glob("*.npz")):
        if "humanact12" in source.stem.lower():
            continue
        target = artifact_root / f"{source.stem}.npz"
        info = process_file(source, target, target_fps=args.target_fps)
        records.append({
            "name": source.stem,
            "split": args.split,
            "artifact": str(target.relative_to(output)),
            "frames": info["frames"],
            "contract_version": CONTRACT_VERSION,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    if not records:
        raise RuntimeError("no valid native-rotation NPZ files were found")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    with manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {len(records)} strict4 artifacts to {manifest}")


if __name__ == "__main__":
    main()
