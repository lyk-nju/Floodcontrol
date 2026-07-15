"""Export one training checkpoint as the formal EMA body tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.vae_wan_1d import BodyVAE
from utils.training.vae.checkpoint import (
    load_ema_checkpoint,
    save_tokenizer_bundle,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    model_config = {
        "latent_dim": args.latent_dim,
        "hidden_dim": args.hidden_dim,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "fps": args.fps,
    }
    model = BodyVAE(
        **model_config,
        motion_stats_path=args.motion_stats,
        require_latent_statistics=False,
    )
    checkpoint_metadata = load_ema_checkpoint(model, args.checkpoint)
    checkpoint_metadata["motion_stats_sha256"] = sha256_file(args.motion_stats)
    recorded_hash_path = Path(args.checkpoint).with_name("ckpt_hash.txt")
    if recorded_hash_path.is_file():
        recorded = json.loads(recorded_hash_path.read_text())
        if int(recorded.get("step", -1)) == checkpoint_metadata["global_step"]:
            if recorded.get("hash") != checkpoint_metadata["training_ema_tensor_sha256"]:
                raise ValueError("EMA tokenizer hash does not match ckpt_hash.txt")
    save_tokenizer_bundle(
        model,
        args.output,
        model_config=model_config,
        checkpoint_metadata=checkpoint_metadata,
    )
    print(
        f"wrote EMA tokenizer at step {checkpoint_metadata['global_step']} "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
