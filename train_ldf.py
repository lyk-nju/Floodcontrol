"""Training migration guard for the hybrid LDF milestone."""

from __future__ import annotations


TRAINING_MIGRATION_ERROR = (
    "Floodcontrol LDF training is BLOCKED_ON_STRICT4_VAE: the model core is "
    "available and the strict four-frame body VAE code is implemented, but no "
    "frozen native-rotation VAE checkpoint, verified latent statistics, or hybrid "
    "training artifacts are connected."
)


def main() -> None:
    raise RuntimeError(TRAINING_MIGRATION_ERROR)


if __name__ == "__main__":
    main()
