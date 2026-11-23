"""Activation functions and their derivatives, written out by hand (NumPy)."""

from __future__ import annotations

import math

import numpy as np

_C = math.sqrt(2.0 / math.pi)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def act_fn(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return sigmoid(z)
    if name == "gelu":  # tanh approximation
        return 0.5 * z * (1.0 + np.tanh(_C * (z + 0.044715 * z**3)))
    raise ValueError(f"unknown activation {name!r}")


def act_grad(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    """d act(z) / dz, given the pre-activation z and the output a = act(z)."""
    if name == "relu":
        return (z > 0).astype(z.dtype)
    if name == "tanh":
        return 1.0 - a * a
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "gelu":
        t = np.tanh(_C * (z + 0.044715 * z**3))
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * _C * (1.0 + 3 * 0.044715 * z * z)
    raise ValueError(f"unknown activation {name!r}")
