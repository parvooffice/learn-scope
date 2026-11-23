"""Perceptually uniform viridis and a blue/red diverging scale, vectorised."""

from __future__ import annotations

import numpy as np

_VIRIDIS = np.array(
    [
        [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
        [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
    ],
    dtype=np.float64,
)


def viridis(t: np.ndarray) -> np.ndarray:
    """t in [0, 1] (any shape) -> uint8 RGB with a trailing axis of 3."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0) * (len(_VIRIDIS) - 1)
    i = np.minimum(np.floor(t).astype(int), len(_VIRIDIS) - 2)
    f = (t - i)[..., None]
    return (_VIRIDIS[i] + (_VIRIDIS[i + 1] - _VIRIDIS[i]) * f).astype(np.uint8)


def diverging(t: np.ndarray, neutral=(235, 238, 242)) -> np.ndarray:
    """t in [-1, 1]: negative = blue, positive = red, 0 = neutral grey."""
    t = np.clip(np.asarray(t, dtype=np.float64), -1.0, 1.0)
    neg, pos = np.array([60, 120, 255.0]), np.array([240, 80, 70.0])
    neutral_a = np.array(neutral, dtype=np.float64)
    target = np.where((t < 0)[..., None], neg, pos)
    return (neutral_a + (target - neutral_a) * np.abs(t)[..., None]).astype(np.uint8)


def rgb_hex(c) -> str:
    r, g, b = (int(v) for v in c)
    return f"#{r:02x}{g:02x}{b:02x}"
