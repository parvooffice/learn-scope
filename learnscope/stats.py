"""Turn raw numbers (weights, gradients, activations) of one layer into compact statistics."""

from __future__ import annotations

import numpy as np

from .snapshots import Hist, LayerStats

HIST_BINS = 24


def histogram(values: np.ndarray, bins: int = HIST_BINS, lo: float | None = None, hi: float | None = None) -> Hist:
    """Fixed-bin histogram. If lo/hi are omitted they come from the data."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    mn = lo if lo is not None else (float(v.min()) if v.size else 0.0)
    mx = hi if hi is not None else (float(v.max()) if v.size else 1.0)
    if not (np.isfinite(mn) and np.isfinite(mx)):
        mn, mx = 0.0, 1.0
    if mx - mn < 1e-12:
        mn, mx = mn - 0.5, mx + 0.5
    width = (mx - mn) / bins
    idx = np.clip(np.floor((v - mn) / width).astype(np.int64), 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).tolist() if v.size else [0] * bins
    return Hist(counts=counts, lo=float(mn), hi=float(mx))


def build_layer_stats(activation: str, nout: int, W: np.ndarray, grad_w: np.ndarray, acts: np.ndarray) -> LayerStats:
    """`acts` has shape (samples, nout)."""
    W = np.asarray(W, dtype=np.float64)
    g = np.asarray(grad_w, dtype=np.float64)
    a = np.asarray(acts, dtype=np.float64)
    w_max = float(np.max(np.abs(W))) if W.size else 0.0
    g_max = float(np.max(np.abs(g))) if g.size else 0.0

    unit_mean = a.mean(axis=0) if a.shape[0] else np.zeros(nout)
    unit_max = a.max(axis=0) if a.shape[0] else np.zeros(nout)
    total = max(1, a.size)
    zero = float(np.count_nonzero(a <= 0)) / total
    sat = 0.0
    if activation == "tanh":
        sat = float(np.count_nonzero(np.abs(a) > 0.95)) / total
    elif activation == "sigmoid":
        sat = float(np.count_nonzero((a < 0.05) | (a > 0.95))) / total
    dead = 0.0
    if activation in ("relu", "gelu"):
        dead = float(np.count_nonzero(unit_max < 1e-3)) / max(1, nout)

    if activation == "tanh":
        a_hist = histogram(a, HIST_BINS, -1.0, 1.0)
    elif activation == "sigmoid":
        a_hist = histogram(a, HIST_BINS, 0.0, 1.0)
    else:
        lo = min(0.0, float(a.min())) if a.size else 0.0
        hi = float(a.max()) if a.size else 1.0
        a_hist = histogram(a, HIST_BINS, lo, hi if hi > lo else lo + 1.0)

    return LayerStats(
        w_mean=float(W.mean()) if W.size else 0.0,
        w_std=float(W.std()) if W.size else 0.0,
        w_abs_max=w_max,
        w_hist=histogram(W, HIST_BINS, -(w_max or 1.0), w_max or 1.0),
        g_mean_abs=float(np.mean(np.abs(g))) if g.size else 0.0,
        g_norm=float(np.sqrt(np.sum(g * g))),
        g_std=float(g.std()) if g.size else 0.0,
        g_hist=histogram(g, HIST_BINS, -(g_max or 1e-12), g_max or 1e-12),
        a_mean=float(a.mean()) if a.size else 0.0,
        a_std=float(a.std()) if a.size else 0.0,
        dead_pct=dead,
        zero_pct=zero,
        sat_pct=sat,
        a_hist=a_hist,
        unit_mean=[float(x) for x in unit_mean],
    )
