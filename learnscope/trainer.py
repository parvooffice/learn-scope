"""The playground trainer: a mini-batch training loop that records full telemetry on demand."""

from __future__ import annotations

import time

import numpy as np

from .config import PlaygroundConfig, layer_specs
from .datasets import Dataset
from .mlp import MLP, Optimizer
from .snapshots import Snapshot
from .stats import build_layer_stats


class Trainer:
    def __init__(self, cfg: PlaygroundConfig, train: Dataset, test: Dataset) -> None:
        self.cfg = cfg
        self.specs = layer_specs(cfg)
        self.net = MLP(self.specs, cfg.seed)
        self.train, self.test = train, test
        self.lr = cfg.lr
        self.l2 = cfg.l2
        self.batch_size = min(cfg.batch_size, train.n)
        self.step = 0
        self._t0 = time.perf_counter()
        self._rng = np.random.default_rng(cfg.seed * 31 + 1)
        self._perm = self._rng.permutation(train.n)
        self._pos = 0
        self._params = [*self.net.W, *self.net.b]  # views: the optimizer updates them in place
        self.opt = Optimizer(cfg.optimizer, self._params)

    # ----------------------------------------------------------------- live settings
    def set_live(self, lr: float | None = None, l2: float | None = None, batch_size: int | None = None) -> None:
        if lr is not None:
            self.lr = float(lr)
        if l2 is not None:
            self.l2 = float(l2)
        if batch_size is not None:
            self.batch_size = max(1, min(int(batch_size), self.train.n))

    # ----------------------------------------------------------------- training
    def _next_batch(self) -> np.ndarray:
        k = self.batch_size
        if self._pos + k > len(self._perm):
            self._perm = self._rng.permutation(self.train.n)
            self._pos = 0
        idx = self._perm[self._pos : self._pos + k]
        self._pos += k
        return idx

    def train_step(self) -> float:
        idx = self._next_batch()
        loss, dW, db = self.net.loss_and_grads(self.train.x[idx], self.train.y[idx], self.l2)
        self.opt.step(self._params, [*dW, *db], self.lr)
        self.step += 1
        return loss

    # ----------------------------------------------------------------- telemetry
    def snapshot(self) -> Snapshot:
        """Metrics, full-batch gradients and per-layer statistics (costs about one extra step)."""
        net = self.net
        tr_loss, tr_acc = net.evaluate(self.train.x, self.train.y)
        te_loss, te_acc = net.evaluate(self.test.x, self.test.y)
        _, dW, _ = net.loss_and_grads(self.train.x, self.train.y, self.l2)
        _, _, acts = net.forward(self.train.x)
        layers = [
            build_layer_stats(s.activation, s.nout, net.W[l], dW[l], acts[l]) for l, s in enumerate(self.specs)
        ]
        return Snapshot(
            step=self.step,
            epoch=self.step * self.batch_size / self.train.n,
            train_loss=tr_loss,
            test_loss=te_loss,
            train_acc=tr_acc,
            test_acc=te_acc,
            lr=self.lr,
            t=time.perf_counter() - self._t0,
            weights=net.weights(),
            layers=layers,
        )
