"""Training migration guard for the hybrid LDF milestone."""

from __future__ import annotations


TRAINING_MIGRATION_ERROR = (
    "Floodcontrol LDF training is BLOCKED_ON_STRICT4_VAE: the model core is "
    "available, but the strict four-frame body VAE, hybrid dataset fields, real "
    "root/local statistics, and root/latent velocity losses are not connected."
)


def main() -> None:
    raise RuntimeError(TRAINING_MIGRATION_ERROR)


if __name__ == "__main__":
    main()
