"""Training migration guard for the hybrid LDF milestone."""

from __future__ import annotations


TRAINING_MIGRATION_ERROR = (
    "Floodcontrol LDF training is BLOCKED_ON_BODY_VAE: the model core is "
    "available and the four-frame body VAE/EMA tokenizer is implemented, "
    "but the frozen online encoder, verified latent statistics, causal "
    "encoder-context sampler, and hybrid training batch are not connected."
)


def main() -> None:
    raise RuntimeError(TRAINING_MIGRATION_ERROR)


if __name__ == "__main__":
    main()
