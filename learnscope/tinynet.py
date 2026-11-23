"""A small neural network built from autograd `Value`s (used to train XOR from scratch)."""

from __future__ import annotations

import numpy as np

from .autograd import Value


def _activate(z: Value, act: str) -> Value:
    if act == "tanh":
        return z.tanh()
    if act == "relu":
        return z.relu()
    if act == "sigmoid":
        return z.sigmoid()
    return z  # linear


class Neuron:
    def __init__(self, nin: int, act: str, rng: np.random.Generator) -> None:
        self.act = act
        self.w = [Value(float(rng.random() * 2 - 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x: list[Value]) -> Value:
        z = self.b
        for wi, xi in zip(self.w, x):
            z = z + wi * xi
        return _activate(z, self.act)

    def params(self) -> list[Value]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, nin: int, nout: int, act: str, rng: np.random.Generator) -> None:
        self.neurons = [Neuron(nin, act, rng) for _ in range(nout)]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]

    def params(self) -> list[Value]:
        return [p for n in self.neurons for p in n.params()]


class MLP:
    def __init__(self, sizes: list[int], acts: list[str], seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.layers = [Layer(sizes[i], sizes[i + 1], acts[i], rng) for i in range(len(sizes) - 1)]

    def __call__(self, x) -> list[Value]:
        a = [Value.of(e) for e in x]
        for layer in self.layers:
            a = layer(a)
        return a

    def params(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.params()]

    def zero_grad(self) -> None:
        for p in self.params():
            p.grad = 0.0


def train_mse(net: MLP, xs, ys, lr: float, steps: int) -> list[float]:
    """Mean squared error + plain SGD. Returns the loss history."""
    history: list[float] = []
    for _ in range(steps):
        loss = Value(0.0)
        for x, y in zip(xs, ys):
            loss = loss + (net(x)[0] - y) ** 2
        loss = loss * (1.0 / len(xs))
        net.zero_grad()
        loss.backward()
        for p in net.params():
            p.data -= lr * p.grad
        history.append(loss.data)
    return history
