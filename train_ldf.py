"""Training migration guard for the hybrid LDF milestone."""

from __future__ import annotations


TRAINING_MIGRATION_ERROR = (
    "Floodcontrol LDF training is BLOCKED_ON_BODY_VAE: the model core is "
    "available, the EMA tokenizer and verified latent statistics are ready, "
    "and BodyVAE.encode_window exposes the causal context boundary; however, "
    "the context-aware crop/collate, frozen online encoder call, and hybrid "
    "training batch are not connected to an LDF Lightning module."
)


def main() -> None:
    raise RuntimeError(TRAINING_MIGRATION_ERROR)


if __name__ == "__main__":
    main()
