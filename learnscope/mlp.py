"""A multi-layer perceptron with hand-written backpropagation (NumPy).

Binary classification: hidden layers use the chosen activations; the output is ONE logit passed
through a sigmoid. Loss = mean binary cross-entropy (+ optional L2 on the weights).
"""

from __future__ import annotations

import numpy as np

from .activations import act_fn, act_grad, sigmoid
from .config import LayerSpec

Weights = list[tuple[np.ndarray, np.ndarray]]


class MLP:
    def __init__(self, specs: list[LayerSpec], seed: int = 0) -> None:
        self.specs = specs
        rng = np.random.default_rng(seed * 977 + 3)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for s in specs:
            # He init for ReLU/GELU, Glorot/Xavier for tanh/sigmoid and the output layer
            std = np.sqrt(2.0 / s.nin) if s.activation in ("relu", "gelu") else np.sqrt(2.0 / (s.nin + s.nout))
            self.W.append(rng.normal(size=(s.nin, s.nout)) * std)
            self.b.append(np.zeros(s.nout))

    # -------------------------------------------------------------------------- forward
    def forward(self, X: np.ndarray):
        """Returns (logits (n,1), zs, acts). `acts[-1]` is the sigmoid of the logits."""
        a = X
        zs: list[np.ndarray] = []
        acts: list[np.ndarray] = []
        last = len(self.specs) - 1
        for l in range(last):
            z = a @ self.W[l] + self.b[l]
            a = act_fn(self.specs[l].activation, z)
            zs.append(z)
            acts.append(a)
        logits = a @ self.W[last] + self.b[last]
        acts.append(sigmoid(logits))
        return logits, zs, acts

    def weights(self) -> Weights:
        return [(w.copy(), b.copy()) for w, b in zip(self.W, self.b)]

    # -------------------------------------------------------------------------- loss
    @staticmethod
    def bce(logits: np.ndarray, y: np.ndarray) -> float:
        """Binary cross-entropy with logits, written stably: softplus(z) - y*z."""
        return float(np.mean(np.logaddexp(0.0, logits[:, 0]) - y * logits[:, 0]))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        logits, _, _ = self.forward(X)
        acc = float(np.mean((logits[:, 0] > 0) == (y > 0.5)))
        return self.bce(logits, y), acc

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray, l2: float = 0.0):
        """Mean loss (including the L2 penalty) and its gradients dW, db for every layer."""
        n = X.shape[0]
        logits, zs, acts = self.forward(X)
        loss = self.bce(logits, y)
        if l2 > 0:
            loss += 0.5 * l2 * sum(float(np.sum(w * w)) for w in self.W)

        dW: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        db: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
        dz = (acts[-1] - y[:, None]) / n  # d(mean BCE)/d(logit) for sigmoid + cross-entropy
        for l in range(len(self.W) - 1, -1, -1):
            a_prev = X if l == 0 else acts[l - 1]
            dW[l] = a_prev.T @ dz + l2 * self.W[l]
            db[l] = dz.sum(axis=0)
            if l > 0:
                da = dz @ self.W[l].T
                dz = da * act_grad(self.specs[l - 1].activation, zs[l - 1], acts[l - 1])
        return loss, dW, db


class Optimizer:
    """SGD, SGD with momentum, and Adam, written out by hand."""

    def __init__(self, kind: str, params: list[np.ndarray]) -> None:
        if kind not in ("sgd", "momentum", "adam"):
            raise ValueError(f"unknown optimizer {kind!r}")
        self.kind = kind
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray], lr: float) -> None:
        self.t += 1
        if self.kind == "sgd":
            for p, g in zip(params, grads):
                p -= lr * g
        elif self.kind == "momentum":
            for p, g, m in zip(params, grads, self.m):
                m *= 0.9
                m += g
                p -= lr * m
        else:
            b1, b2, eps = 0.9, 0.999, 1e-8
            c1 = 1 - b1**self.t
            c2 = 1 - b2**self.t
            for p, g, m, v in zip(params, grads, self.m, self.v):
                m *= b1
                m += (1 - b1) * g
                v *= b2
                v += (1 - b2) * g * g
                p -= lr * (m / c1) / (np.sqrt(v / c2) + eps)
