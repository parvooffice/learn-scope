from __future__ import annotations

import math

import numpy as np

from ..snapshots import Snapshot


def last(a: list, k: int = 1) -> list:
    return a[max(0, len(a) - k) :]


def avg(a: list[float]) -> float:
    return sum(a) / len(a) if a else 0.0


def nice_lr(x: float) -> float:
    """Round a learning rate to 2 significant digits so suggestions look human (0.01, not 0.0100003)."""
    return float(f"{min(3.0, max(0.0001, x)):.2g}")


def is_bad(n: float) -> bool:
    return not math.isfinite(n)


def snapshot_has_nan(s: Snapshot) -> bool:
    if is_bad(s.train_loss) or is_bad(s.test_loss):
        return True
    return any(not np.isfinite(w).all() for w, _ in s.weights)


def fmt(x: float, digits: int = 3) -> str:
    """Compact number formatting for explanations (mirrors JavaScript's toPrecision / toExponential)."""
    if not math.isfinite(x):
        return str(x)
    if abs(x) < 1e-3 and x != 0:
        s = f"{x:.1e}"
        mant, exp = s.split("e")
        return f"{mant}e{int(exp)}"
    return f"{float(f'{x:.{digits}g}'):g}"
