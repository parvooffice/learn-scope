"""Loss landscape slice (after Li et al., "Visualizing the Loss Landscape of Neural Nets").

The loss depends on ALL weights: far too many dimensions to draw. We pick two random orthogonal
directions d1, d2 in weight space and plot the loss on the plane
        w(alpha, beta) = w_center + alpha*d1 + beta*d2
Directions are scaled layer by layer to the size of that layer's weights, so "alpha = 1" means
"change every layer by about 100% of its own size": slices are comparable across networks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import LayerSpec
from .forward import Weights, evaluate


def flatten(w: Weights) -> np.ndarray:
    return np.concatenate([np.concatenate([W.ravel(), b.ravel()]) for W, b in w])


def unflatten(template: Weights, flat: np.ndarray) -> Weights:
    out: Weights = []
    o = 0
    for W, b in template:
        nw, nb = W.size, b.size
        out.append((flat[o : o + nw].reshape(W.shape).copy(), flat[o + nw : o + nw + nb].copy()))
        o += nw + nb
    return out


def _layer_ranges(template: Weights) -> list[tuple[int, int]]:
    ranges, o = [], 0
    for W, b in template:
        ranges.append((o, o + W.size + b.size))
        o += W.size + b.size
    return ranges


def random_directions(center: Weights, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two orthogonal random directions, each rescaled per layer to that layer's weight norm."""
    rng = np.random.default_rng(seed * 6151 + 17)
    c = flatten(center)
    ranges = _layer_ranges(center)

    def normalise(d: np.ndarray) -> None:
        for s, e in ranges:
            dn = np.linalg.norm(d[s:e])
            cn = np.linalg.norm(c[s:e]) or np.sqrt(e - s)
            d[s:e] *= (cn / dn) if dn > 0 else 0.0

    d1 = rng.normal(size=c.size)
    d2 = rng.normal(size=c.size)
    d2 -= (d2 @ d1) / (d1 @ d1) * d1  # Gram-Schmidt: projections onto (d1, d2) are exact coordinates
    normalise(d1)
    normalise(d2)
    d2 -= (d2 @ d1) / (d1 @ d1) * d1  # per-layer rescaling slightly breaks orthogonality: fix once more
    return d1, d2


def project(w: np.ndarray, center: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> tuple[float, float]:
    """Coordinates (alpha, beta) of a weight vector in the slice plane."""
    d = w - center
    return float(d @ d1 / (d1 @ d1)), float(d @ d2 / (d2 @ d2))


@dataclass
class Landscape:
    grid: int
    range_: float
    values: np.ndarray  # (grid, grid); row 0 is beta = +range (top of the picture)
    min: float
    clip: float  # 95th percentile: top of the colour scale, so one spike cannot wash it out
    center_loss: float
    d1: np.ndarray
    d2: np.ndarray


def compute_landscape(
    specs: list[LayerSpec],
    center: Weights,
    X: np.ndarray,
    y: np.ndarray,
    grid: int = 31,
    range_: float = 1.0,
    seed: int = 1,
    on_progress: Callable[[float], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Landscape | None:
    """Evaluate the loss on a grid x grid slice around `center` (use an odd grid so the centre cell is exact)."""
    d1, d2 = random_directions(center, seed)
    c = flatten(center)
    values = np.zeros((grid, grid))
    for gy in range(grid):
        beta = range_ - gy / (grid - 1) * 2 * range_
        for gx in range(grid):
            alpha = -range_ + gx / (grid - 1) * 2 * range_
            values[gy, gx] = evaluate(specs, unflatten(center, c + alpha * d1 + beta * d2), X, y)[0]
        if on_progress:
            on_progress((gy + 1) / grid)
        if cancelled and cancelled():
            return None
    flat = np.sort(values.ravel())
    return Landscape(
        grid=grid,
        range_=range_,
        values=values,
        min=float(flat[0]),
        clip=float(flat[int(len(flat) * 0.95)]),
        center_loss=evaluate(specs, center, X, y)[0],
        d1=d1,
        d2=d2,
    )
