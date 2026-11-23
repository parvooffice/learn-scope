"""Synthetic training histories for testing the Model Doctor without training anything."""

from __future__ import annotations

import numpy as np

from learnscope.config import DEFAULT_CONFIG, HiddenLayer, PlaygroundConfig
from learnscope.snapshots import Hist, LayerStats, Snapshot


def empty_hist() -> Hist:
    return Hist([0] * 24, 0.0, 1.0)


def mk_layer(**over) -> LayerStats:
    base = dict(
        w_mean=0.0, w_std=0.5, w_abs_max=1.0, w_hist=empty_hist(), g_mean_abs=0.05, g_norm=0.2, g_std=0.05,
        g_hist=empty_hist(), a_mean=0.1, a_std=0.4, dead_pct=0.0, zero_pct=0.3, sat_pct=0.05, a_hist=empty_hist(),
        unit_mean=[0.1],
    )
    base.update(over)
    return LayerStats(**base)


def mk_snap(step: int, cfg: PlaygroundConfig, layer_over=None, **over) -> Snapshot:
    n = len(cfg.layers) + 1
    base = dict(
        step=step, epoch=step / 10, train_loss=0.5, test_loss=0.5, train_acc=0.8, test_acc=0.8, lr=cfg.lr, t=step * 0.01,
        weights=[(np.zeros((2, 2)), np.zeros(2)) for _ in range(n)],
        layers=[mk_layer(**(layer_over(i) if layer_over else {})) for i in range(n)],
    )
    base.update(over)
    return Snapshot(**base)


def mk_history(cfg: PlaygroundConfig, count: int, f) -> list[Snapshot]:
    """`count` snapshots (step = 5 * (index + 1)); f(i) returns overrides for snapshot i."""
    out = []
    for i in range(count):
        over = dict(f(i))
        layer_over = over.pop("layer_over", None)
        out.append(mk_snap((i + 1) * 5, cfg, layer_over=layer_over, **over))
    return out


def cfg_with(**over) -> PlaygroundConfig:
    return DEFAULT_CONFIG.with_(**over)


def layers(units: int, act: str, count: int) -> tuple[HiddenLayer, ...]:
    return tuple(HiddenLayer(units, act) for _ in range(count))
