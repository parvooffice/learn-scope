"""Training telemetry: what is recorded at every snapshot, and the bounded history that keeps it."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class Hist:
    counts: list[int]
    lo: float
    hi: float


@dataclass
class LayerStats:
    # weights
    w_mean: float
    w_std: float
    w_abs_max: float
    w_hist: Hist
    # gradient of the loss w.r.t. this layer's weights (full train batch)
    g_mean_abs: float
    g_norm: float
    g_std: float
    g_hist: Hist
    # activations (after the activation function) on the training set
    a_mean: float
    a_std: float
    dead_pct: float  # fraction of units that never fire on any sample (relu / gelu)
    zero_pct: float  # fraction of activations that are exactly <= 0
    sat_pct: float  # fraction stuck near the flat ends (tanh / sigmoid)
    a_hist: Hist
    unit_mean: list[float] = field(default_factory=list)  # mean activation per unit


@dataclass
class Snapshot:
    step: int
    epoch: float
    train_loss: float
    test_loss: float
    train_acc: float
    test_acc: float
    lr: float
    t: float  # seconds since the run started
    weights: list[tuple[np.ndarray, np.ndarray]]  # per layer: (W[nin, nout], b[nout])
    layers: list[LayerStats]


def snapshot_bytes(s: Snapshot) -> int:
    n = 256
    for w, b in s.weights:
        n += w.nbytes + b.nbytes
    for layer in s.layers:
        n += 8 * (3 * 24 + 16 + len(layer.unit_mean))
    return n


class HistoryBuffer(Generic[T]):
    """A bounded history with two limits: item count and bytes.

    When either is exceeded we do NOT just drop the oldest item (that would make time travel forget
    how training began). We thin out the OLDEST HALF by keeping every second item instead, so old
    history gets coarser while recent history stays dense. First and last items are always kept.
    """

    def __init__(self, size_of: Callable[[T], int], max_items: int = 4000, max_bytes: int = 150 * 1024 * 1024) -> None:
        self._size_of = size_of
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._items: list[T] = []
        self._sizes: list[int] = []
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def total_bytes(self) -> int:
        return self._bytes

    def at(self, i: int) -> T | None:
        return self._items[i] if 0 <= i < len(self._items) else None

    def last(self) -> T | None:
        return self._items[-1] if self._items else None

    def to_list(self) -> list[T]:
        return list(self._items)

    def push(self, item: T) -> None:
        size = self._size_of(item)
        self._items.append(item)
        self._sizes.append(size)
        self._bytes += size
        if len(self._items) > self.max_items or self._bytes > self.max_bytes:
            self._thin()

    def clear(self) -> None:
        self._items, self._sizes, self._bytes = [], [], 0

    def _thin(self) -> None:
        half = len(self._items) // 2
        keep = [i for i in range(len(self._items)) if i >= half or i % 2 == 0]
        self._items = [self._items[i] for i in keep]
        self._sizes = [self._sizes[i] for i in keep]
        self._bytes = sum(self._sizes)

    def nearest(self, key: float, key_of: Callable[[T], float]) -> int:
        """Index of the item whose key is closest to `key` (items must be ordered by key). -1 if empty."""
        n = len(self._items)
        if n == 0:
            return -1
        lo = bisect_left(self._items, key, key=key_of)
        if lo >= n:
            return n - 1
        if lo > 0 and abs(key_of(self._items[lo - 1]) - key) <= abs(key_of(self._items[lo]) - key):
            return lo - 1
        return lo
