"""The shipped MNIST subset: 10,000 training and 2,000 test images (learnscope/data/mnist-12k.npz)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "mnist-12k.npz"


@dataclass
class MnistData:
    train_x: np.ndarray  # (10000, 1, 28, 28) float32 in [0, 1]
    train_y: np.ndarray  # (10000,) int64
    test_x: np.ndarray  # (2000, 1, 28, 28)
    test_y: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train_y.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test_y.shape[0])


@lru_cache(maxsize=1)
def load_mnist(path: str | None = None) -> MnistData:
    with np.load(path or DATA_FILE) as f:
        images, labels, n_train = f["images"], f["labels"].astype(np.int64), int(f["n_train"])
    x = (images.astype(np.float32) / 255.0)[:, None, :, :]
    return MnistData(x[:n_train], labels[:n_train], x[n_train:], labels[n_train:])
