"""The 2-3-1 network behind the Backprop explainer.

    z_j = w_j1*x1 + w_j2*x2 + b_j        (3 hidden neurons)
    h_j = tanh(z_j)
    yhat = v1*h1 + v2*h2 + v3*h3 + c
    L = (yhat - t)^2

Every number the explainer shows is computed here, live, by the autograd engine, and cross-checked
against the chain rule written out by hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .autograd import Value


@dataclass(frozen=True)
class ExplainerParams:
    x: tuple[float, float] = (0.5, -0.8)
    t: float = 0.7
    w: tuple[tuple[float, float], ...] = ()  # w[j][i]: input i -> hidden j
    b: tuple[float, ...] = (0.1, -0.1, 0.05)
    v: tuple[float, ...] = ()
    c: float = 0.0


@dataclass
class Grads:
    yhat: float
    v: list[float]
    c: float
    h: list[float]
    z: list[float]
    w: list[list[float]]
    b: list[float]

    def flat(self) -> list[float]:
        return [self.yhat, self.c, *self.v, *self.h, *self.z, *self.b, *[g for row in self.w for g in row]]


@dataclass
class ExplainerResult:
    z: list[float]
    h: list[float]
    yhat: float
    loss: float
    auto: Grads  # from Value.backward()
    manual: Grads  # the same, written out by hand with the chain rule
    max_diff: float = field(default=0.0)  # largest |auto - manual|; should be ~1e-16


def initial_params(seed: int = 4) -> ExplainerParams:
    rng = np.random.default_rng(seed)
    r = lambda: round(float(rng.random() * 2 - 1), 2)  # noqa: E731
    return ExplainerParams(
        x=(0.5, -0.8),
        t=0.7,
        w=((r(), r()), (r(), r()), (r(), r())),
        b=(0.1, -0.1, 0.05),
        v=(r(), r(), r()),
        c=0.0,
    )


def run_explainer(p: ExplainerParams) -> ExplainerResult:
    x = [Value(e, label=f"x{i + 1}") for i, e in enumerate(p.x)]
    w = [[Value(e, label=f"w{j + 1}{i + 1}") for i, e in enumerate(row)] for j, row in enumerate(p.w)]
    b = [Value(e, label=f"b{j + 1}") for j, e in enumerate(p.b)]
    vv = [Value(e, label=f"v{j + 1}") for j, e in enumerate(p.v)]
    c = Value(p.c, label="c")

    z = [row[0] * x[0] + row[1] * x[1] + b[j] for j, row in enumerate(w)]
    h = [zj.tanh() for zj in z]
    yhat = vv[0] * h[0] + vv[1] * h[1] + vv[2] * h[2] + c
    loss = (yhat - p.t) ** 2
    loss.backward()

    auto = Grads(
        yhat=yhat.grad,
        v=[e.grad for e in vv],
        c=c.grad,
        h=[e.grad for e in h],
        z=[e.grad for e in z],
        w=[[e.grad for e in row] for row in w],
        b=[e.grad for e in b],
    )

    # the chain rule, by hand
    hv = [e.data for e in h]
    dyhat = 2 * (yhat.data - p.t)
    dh = [dyhat * vj for vj in p.v]
    dz = [d * (1 - hv[j] * hv[j]) for j, d in enumerate(dh)]
    manual = Grads(
        yhat=dyhat,
        v=[dyhat * hj for hj in hv],
        c=dyhat,
        h=dh,
        z=dz,
        w=[[d * p.x[0], d * p.x[1]] for d in dz],
        b=list(dz),
    )
    max_diff = max(abs(a - m) for a, m in zip(auto.flat(), manual.flat()))
    return ExplainerResult([e.data for e in z], hv, yhat.data, loss.data, auto, manual, max_diff)


def apply_step(p: ExplainerParams, g: Grads, lr: float) -> ExplainerParams:
    """One gradient-descent step: param <- param - lr * gradient."""
    return replace(
        p,
        w=tuple(tuple(e - lr * g.w[j][i] for i, e in enumerate(row)) for j, row in enumerate(p.w)),
        b=tuple(e - lr * g.b[j] for j, e in enumerate(p.b)),
        v=tuple(e - lr * g.v[j] for j, e in enumerate(p.v)),
        c=p.c - lr * g.c,
    )
