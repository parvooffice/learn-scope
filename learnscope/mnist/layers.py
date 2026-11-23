"""Neural-network layers with hand-written forward AND backward passes (NumPy).

Convolution is done as im2col + matrix multiply: every receptive field becomes a row, so the
whole convolution is one big matmul. Backward scatters gradients back with a loop over kernel
positions ("col2im").
"""

from __future__ import annotations

import numpy as np


class Layer:
    params: list[np.ndarray]
    grads: list[np.ndarray]

    def __init__(self) -> None:
        self.params, self.grads = [], []

    def forward(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class Conv2D(Layer):
    """Valid convolution, stride 1. x: (N, C, H, W) -> (N, O, H-KH+1, W-KW+1)."""

    def __init__(self, c_in: int, c_out: int, k: int, rng: np.random.Generator, dtype=np.float32) -> None:
        super().__init__()
        self.k, self.c_in, self.c_out = k, c_in, c_out
        fan_in = c_in * k * k
        self.W = (rng.normal(size=(c_out, fan_in)) * np.sqrt(2.0 / fan_in)).astype(dtype)  # He init
        self.b = np.zeros(c_out, dtype=dtype)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        k = self.k
        oh, ow = h - k + 1, w - k + 1
        win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))  # (n, c, oh, ow, k, k)
        self._cols = np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5)).reshape(n * oh * ow, c * k * k)
        self._shape = (n, c, h, w, oh, ow)
        out = self._cols @ self.W.T + self.b  # (n*oh*ow, o)
        return np.ascontiguousarray(out.reshape(n, oh, ow, self.c_out).transpose(0, 3, 1, 2))

    def backward(self, dout: np.ndarray) -> np.ndarray:
        n, c, h, w, oh, ow = self._shape
        k = self.k
        d2 = dout.transpose(0, 2, 3, 1).reshape(-1, self.c_out)
        self.grads[0][...] = d2.T @ self._cols
        self.grads[1][...] = d2.sum(axis=0)
        dcols = (d2 @ self.W).reshape(n, oh, ow, c, k, k)
        dx = np.zeros((n, c, h, w), dtype=dout.dtype)
        for i in range(k):
            for j in range(k):
                dx[:, :, i : i + oh, j : j + ow] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx


class MaxPool2(Layer):
    """2x2 max pooling, stride 2 (height and width must be even)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        if h % 2 or w % 2:
            raise ValueError("MaxPool2 needs even height and width")
        r = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = r.max(axis=(3, 5))
        self._mask = r == out[:, :, :, None, :, None]
        self._shape = x.shape
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        mask = self._mask
        share = mask / mask.sum(axis=(3, 5), keepdims=True)  # ties split the gradient
        return (share * dout[:, :, :, None, :, None]).reshape(self._shape)


class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pos = x > 0
        return x * self._pos

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._pos


class Flatten(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, dtype=np.float32) -> None:
        super().__init__()
        self.W = (rng.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.grads[0][...] = self._x.T @ dout
        self.grads[1][...] = dout.sum(axis=0)
        return dout @ self.W.T


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient with respect to the logits: (p - onehot) / N."""
    p = softmax(logits)
    n = logits.shape[0]
    loss = float(-np.mean(np.log(np.maximum(p[np.arange(n), y], 1e-12))))
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3) -> None:
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1**self.t, 1 - b2**self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + eps)).astype(p.dtype)
