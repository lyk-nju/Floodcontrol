"""HumanML22 skeleton topology used by motion visualization."""

from __future__ import annotations


HUMANML22_CHAINS: tuple[tuple[int, ...], ...] = (
    (0, 2, 5, 8, 11),
    (0, 1, 4, 7, 10),
    (0, 3, 6, 9, 12, 15),
    (9, 14, 17, 19, 21),
    (9, 13, 16, 18, 20),
)

# One RGB color per kinematic chain. Values are uint8-ready and intentionally
# shared by original and reconstruction videos.
HUMANML22_CHAIN_COLORS: tuple[tuple[int, int, int], ...] = (
    (254, 178, 26),
    (0, 170, 255),
    (19, 70, 134),
    (255, 182, 0),
    (0, 212, 126),
)


__all__ = ["HUMANML22_CHAINS", "HUMANML22_CHAIN_COLORS"]
