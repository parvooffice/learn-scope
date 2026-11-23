"""2-D toy datasets (all points lie roughly inside [-1.2, 1.2]^2). Deterministic for a given seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    x: np.ndarray  # (n, 2) float64
    y: np.ndarray  # (n,) float64 in {0, 1}

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


def generate_dataset(kind: str, n: int, noise: float, seed: int) -> Dataset:
    rng = np.random.default_rng(seed * 7919 + 13)
    cls = np.arange(n) % 2
    y = cls.astype(np.float64)

    def jitter(scale: float = 1.0) -> np.ndarray:
        return rng.normal(size=(n, 2)) * noise * scale

    if kind == "moons":
        t = rng.random(n) * np.pi
        px = np.where(cls == 0, np.cos(t), 1 - np.cos(t))
        py = np.where(cls == 0, np.sin(t), 0.5 - np.sin(t))
        j = jitter(0.5)
        x = np.stack([(px - 0.5) * 0.85 + j[:, 0], (py - 0.25) * 1.3 + j[:, 1]], axis=1)
    elif kind == "circles":
        r = np.where(cls == 0, 0.42, 0.88)
        t = rng.random(n) * 2 * np.pi
        j = jitter(0.5)
        x = np.stack([r * np.cos(t) + j[:, 0], r * np.sin(t) + j[:, 1]], axis=1)
    elif kind == "spiral":
        t = 0.08 + rng.random(n) * 0.92
        theta = t * 1.75 * 2 * np.pi + cls * np.pi
        j = jitter(0.5)
        x = np.stack([t * 0.95 * np.cos(theta) + j[:, 0], t * 0.95 * np.sin(theta) + j[:, 1]], axis=1)
    elif kind == "xor":
        px = rng.random(n) * 2 - 1
        py = rng.random(n) * 2 - 1
        px = px + np.sign(px) * 0.08  # keep points away from the axes so the problem is learnable
        py = py + np.sign(py) * 0.08
        y = ((px > 0) != (py > 0)).astype(np.float64)
        j = jitter(0.6)
        x = np.stack([px + j[:, 0], py + j[:, 1]], axis=1)
    elif kind == "blobs":
        c = np.where(cls == 0, -0.5, 0.5)
        spread = 0.12 + noise * 0.8
        x = np.stack([c + rng.normal(size=n) * spread, c + rng.normal(size=n) * spread], axis=1)
    else:
        raise ValueError(f"unknown dataset {kind!r}")
    return Dataset(x=x, y=y)


def split_dataset(ds: Dataset, test_frac: float, seed: int) -> tuple[Dataset, Dataset]:
    """Deterministic shuffle-then-split into (train, test)."""
    rng = np.random.default_rng(seed * 104729 + 5)
    idx = rng.permutation(ds.n)
    n_test = max(1, int(round(ds.n * test_frac)))
    test_i, train_i = idx[:n_test], idx[n_test:]
    return Dataset(ds.x[train_i], ds.y[train_i]), Dataset(ds.x[test_i], ds.y[test_i])
