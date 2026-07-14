"""Training migration guard for the hybrid LDF milestone."""

from __future__ import annotations


TRAINING_MIGRATION_ERROR = (
    "Floodcontrol LDF training is BLOCKED_ON_BODY_VAE: the model core is "
    "available and the four-frame body VAE code is implemented, but no "
    "frozen BodyVAE checkpoint, verified latent statistics, or hybrid "
    "training artifacts are connected."
)


def main() -> None:
    raise RuntimeError(TRAINING_MIGRATION_ERROR)


if __name__ == "__main__":
    main()
