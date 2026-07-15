"""EMA-only checkpoint loading and deployable body-tokenizer export."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch

from utils.conditions.vae import CONTRACT_VERSION
from utils.motion_representation import MOTION_CONVERTER_VERSION


TOKENIZER_FORMAT_VERSION = "body-vae-tokenizer-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def tensor_values_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Match the training callback's historical EMA-applied tensor hash."""

    digest = hashlib.sha256()
    for _, value in sorted(state_dict.items()):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def ema_state_dict(model, checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Apply Lightning EMA shadows to all trainable encoder/decoder parameters."""

    if "state_dict" not in checkpoint:
        raise ValueError("VAE checkpoint is missing state_dict")
    if "ema_state" not in checkpoint:
        raise ValueError("VAE checkpoint is missing ema_state; raw weights are forbidden")
    raw_state = checkpoint["state_dict"]
    ema_state = checkpoint["ema_state"]
    if not isinstance(raw_state, Mapping) or not isinstance(ema_state, Mapping):
        raise TypeError("invalid VAE checkpoint state")
    shadows = ema_state.get("shadow_params")
    if not isinstance(shadows, (tuple, list)):
        raise ValueError("VAE ema_state is missing shadow_params")
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if len(trainable_names) != len(shadows):
        raise ValueError(
            "EMA parameter count does not match BodyVAE architecture: "
            f"{len(shadows)} != {len(trainable_names)}"
        )
    inference_state = {
        str(name): value.detach().cpu().clone()
        for name, value in raw_state.items()
        if torch.is_tensor(value)
    }
    for name, shadow in zip(trainable_names, shadows, strict=True):
        if name not in inference_state:
            raise ValueError(f"EMA parameter {name!r} is absent from checkpoint state_dict")
        inference_state[name] = shadow.detach().cpu().clone()
    return inference_state


def load_ema_checkpoint(model, checkpoint_path: str | Path) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    raw_state = checkpoint.get("state_dict", {})
    configured_state = model.state_dict()
    for name in (
        "body_cont_mean", "body_cont_std", "local_root_mean", "local_root_std"
    ):
        if name not in raw_state or not torch.equal(
            raw_state[name].detach().cpu(), configured_state[name].detach().cpu()
        ):
            raise ValueError(
                f"motion statistics do not match the training checkpoint buffer {name!r}"
            )
    inference_state = ema_state_dict(model, checkpoint)
    model.load_state_dict(inference_state, strict=True)
    return {
        "weights_kind": "ema",
        "training_checkpoint_sha256": sha256_file(checkpoint_path),
        "inference_state_sha256": state_dict_sha256(inference_state),
        "training_ema_tensor_sha256": tensor_values_sha256(inference_state),
        "global_step": int(checkpoint.get("global_step", -1)),
    }


def save_tokenizer_bundle(
    model,
    output: str | Path,
    *,
    model_config: Mapping[str, object],
    checkpoint_metadata: Mapping[str, object],
) -> None:
    if checkpoint_metadata.get("weights_kind") != "ema":
        raise ValueError("formal tokenizer bundles require EMA weights")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    inference_hash = state_dict_sha256(state)
    if inference_hash != checkpoint_metadata.get("inference_state_sha256"):
        raise ValueError("model state no longer matches the loaded EMA checkpoint")
    bundle = {
        "format_version": TOKENIZER_FORMAT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "converter_version": MOTION_CONVERTER_VERSION,
        "weights_kind": "ema",
        "model_config": dict(model_config),
        "state_dict": state,
        "metadata": dict(checkpoint_metadata),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output)


def load_tokenizer_bundle(model, path: str | Path) -> dict[str, object]:
    bundle = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if bundle.get("format_version") != TOKENIZER_FORMAT_VERSION:
        raise ValueError("tokenizer bundle format version mismatch")
    if bundle.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("tokenizer bundle contract version mismatch")
    if bundle.get("converter_version") != MOTION_CONVERTER_VERSION:
        raise ValueError("tokenizer bundle converter version mismatch")
    if bundle.get("weights_kind") != "ema":
        raise ValueError("formal tokenizer bundle must contain EMA weights")
    state = bundle.get("state_dict")
    if not isinstance(state, Mapping):
        raise TypeError("tokenizer bundle state_dict is missing")
    model.load_state_dict(state, strict=True)
    metadata = dict(bundle.get("metadata", {}))
    actual = state_dict_sha256(model.state_dict())
    if metadata.get("inference_state_sha256") != actual:
        raise ValueError("tokenizer bundle inference-state hash mismatch")
    return metadata


__all__ = [
    "TOKENIZER_FORMAT_VERSION", "ema_state_dict", "load_ema_checkpoint",
    "load_tokenizer_bundle", "save_tokenizer_bundle", "sha256_file",
    "state_dict_sha256",
    "tensor_values_sha256",
]
