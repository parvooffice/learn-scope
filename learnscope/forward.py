"""Forward passes from a snapshot's stored weights: decision boundary, neuron previews, evaluation.

Because every snapshot stores its weights, any view can be redrawn exactly for any recorded step:
that is what makes time travel work.
"""

from __future__ import annotations

import numpy as np

from .activations import act_fn, sigmoid
from .config import LayerSpec

Weights = list[tuple[np.ndarray, np.ndarray]]

EXTENT = 1.3
GRID = 48


def forward_all(specs: list[LayerSpec], weights: Weights, X: np.ndarray) -> list[np.ndarray]:
    """Every layer's activations for the points X (n, 2). The last entry is the output probability."""
    a = X
    out: list[np.ndarray] = []
    for l, s in enumerate(specs):
        W, b = weights[l]
        z = a @ W + b
        a = sigmoid(z) if l == len(specs) - 1 else act_fn(s.activation, z)
        out.append(a)
    return out


def forward_grid(specs: list[LayerSpec], weights: Weights, g: int = GRID, extent: float = EXTENT) -> list[np.ndarray]:
    """Evaluate every unit of every layer on a g x g grid covering [-extent, extent]^2.

    Result[l] has shape (g, g, nout); row 0 is the TOP of the picture (largest y)."""
    xs = -extent + (np.arange(g) + 0.5) / g * 2 * extent
    ys = extent - (np.arange(g) + 0.5) / g * 2 * extent
    gx, gy = np.meshgrid(xs, ys)  # gy[0] = largest y
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return [a.reshape(g, g, -1) for a in forward_all(specs, weights, pts)]


def evaluate(specs: list[LayerSpec], weights: Weights, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(binary cross-entropy, accuracy) of the network described by `weights`."""
    p = np.clip(forward_all(specs, weights, X)[-1][:, 0], 1e-7, 1 - 1e-7)
    loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return loss, float(np.mean((p > 0.5) == (y > 0.5)))
