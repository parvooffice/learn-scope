"""Plotly figure builders. Pure functions (data in, figure out), so they can be tested without a browser.

Figures are plain Plotly JSON dicts (``{"data": [...], "layout": {...}}``), not ``plotly.graph_objects``
objects: building validated Figure objects costs tens of milliseconds each, which starved the training
thread; a dict costs microseconds and NiceGUI's ``ui.plotly`` accepts it directly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..colors import rgb_hex, viridis
from ..config import LayerSpec
from ..datasets import Dataset
from ..forward import EXTENT
from ..landscape import Landscape
from ..runner import Series
from ..snapshots import Hist, Snapshot

Figure = dict[str, Any]

TEAL, AMBER, VIOLET, RED, BLUE, GREEN, PINK = "#2dd4bf", "#f5b942", "#a78bfa", "#f0625c", "#4f8cff", "#34d399", "#f472b6"
LAYER_COLORS = [TEAL, AMBER, VIOLET, RED, BLUE, GREEN, PINK]
VIRIDIS_SCALE = [[i / 9, rgb_hex(c)] for i, c in enumerate(viridis(np.linspace(0, 1, 10)))]


def _none(x):
    if isinstance(x, list):
        return [_none(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def L(a) -> list:
    """numpy array / sequence -> JSON-safe nested list (NaN and Infinity become null)."""
    return _none(np.asarray(a, dtype=float).tolist())


def _colors(dark: bool) -> tuple[str, str]:
    return ("#8b98a8", "#243040") if dark else ("#5a6778", "#d5dce5")


def _layout(dark: bool, height: int, **extra) -> dict:
    txt, grid = _colors(dark)
    base: dict = dict(
        height=height,
        margin=dict(l=44, r=10, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=txt, size=11),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        xaxis=dict(gridcolor=grid, zeroline=False),
        yaxis=dict(gridcolor=grid, zeroline=False),
        uirevision="keep",  # keep zoom/pan while data updates
    )
    for k, v in extra.items():
        if k in ("xaxis", "yaxis") and isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base


def _fig(data: list, layout: dict) -> Figure:
    return {"data": data, "layout": layout}


def line(x, y, name: str, color: str, width: float = 2, dash: str | None = None, **kw) -> dict:
    ln: dict = dict(color=color, width=width)
    if dash:
        ln["dash"] = dash
    return dict(type="scatter", mode="lines", x=L(x), y=L(y), name=name, line=ln, **kw)


def _marker_shape(step: int | None) -> list[dict]:
    if step is None:
        return []
    return [dict(type="line", x0=step, x1=step, y0=0, y1=1, yref="paper", line=dict(color=TEAL, width=1.5))]


# ------------------------------------------------------------------------------------ training curves
def loss_figure(s: Series, marker_step: int | None, dark: bool = True, height: int = 200) -> Figure:
    return _fig([line(s.steps, s.train_loss, "train", TEAL), line(s.steps, s.test_loss, "test", AMBER)],
                _layout(dark, height, yaxis=dict(rangemode="tozero"), shapes=_marker_shape(marker_step)))


def accuracy_figure(s: Series, marker_step: int | None, dark: bool = True, height: int = 170) -> Figure:
    return _fig([line(s.steps, s.train_acc, "train", TEAL), line(s.steps, s.test_acc, "test", AMBER)],
                _layout(dark, height, yaxis=dict(range=[0, 1.02], tickformat=".0%"), shapes=_marker_shape(marker_step)))


def gradient_figure(s: Series, marker_step: int | None, dark: bool = True, height: int = 200) -> Figure:
    n = s.g_norms.shape[1] if s.g_norms.ndim == 2 else 0
    data = []
    for l in range(n):
        y = np.where(s.g_norms[:, l] > 0, s.g_norms[:, l], np.nan)
        data.append(line(s.steps, y, "out" if l == n - 1 else f"L{l + 1}", LAYER_COLORS[l % len(LAYER_COLORS)], 1.6))
    return _fig(data, _layout(dark, height, yaxis=dict(type="log", exponentformat="e"), shapes=_marker_shape(marker_step)))


def layer_hist_figure(hists: list[Hist], titles: list[str], colors: list[str], dark: bool = True,
                      zero_line: bool = True, row_height: int = 76) -> Figure:
    """One figure with a small histogram per layer (one chart element updates far faster than many)."""
    txt, grid = _colors(dark)
    n = len(hists)
    gap = min(0.12, 0.5 / max(n, 1))
    h = (1 - gap * (n - 1)) / n
    data, layout_axes, annotations, shapes = [], {}, [], []
    for i, hist in enumerate(hists):
        k = len(hist.counts)
        width = (hist.hi - hist.lo) / k
        top = 1 - i * (h + gap)
        sfx = "" if i == 0 else str(i + 1)
        data.append(dict(type="bar", x=L(hist.lo + (np.arange(k) + 0.5) * width), y=list(hist.counts), width=width * 0.92,
                         marker=dict(color=colors[i % len(colors)]), xaxis=f"x{sfx}", yaxis=f"y{sfx}", hoverinfo="skip"))
        layout_axes[f"xaxis{sfx}"] = dict(domain=[0, 1], anchor=f"y{sfx}", gridcolor=grid, zeroline=False)
        layout_axes[f"yaxis{sfx}"] = dict(domain=[top - h, top], anchor=f"x{sfx}", showgrid=False, showticklabels=False)
        annotations.append(dict(text=titles[i], x=0, xref="paper", y=top, yref="paper", xanchor="left", yanchor="bottom",
                                showarrow=False, font=dict(size=10, color=txt)))
        if zero_line and hist.lo < 0 < hist.hi:
            shapes.append(dict(type="line", x0=0, x1=0, y0=0, y1=1, xref=f"x{sfx}", yref=f"y{sfx} domain",
                               line=dict(color=txt, dash="dot", width=1)))
    layout = _layout(dark, n * row_height + 24, margin=dict(l=8, r=8, t=22, b=22), showlegend=False,
                     annotations=annotations, shapes=shapes)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout.update(layout_axes)
    return _fig(data, layout)


# ------------------------------------------------------------------------------------ playground views
def _square_axes(dark: bool, extent: float, **kw) -> dict:
    return dict(
        xaxis=dict(range=[-extent, extent], showgrid=False, zeroline=False, constrain="domain"),
        yaxis=dict(range=[-extent, extent], showgrid=False, zeroline=False, scaleanchor="x", constrain="domain"), **kw)


def boundary_figure(field: np.ndarray | None, train: Dataset, test: Dataset, dark: bool = True, height: int = 340,
                    show_test: bool = True) -> Figure:
    """field[i, j] = P(class 1); row 0 is the TOP of the picture."""
    data: list[dict] = []
    if field is not None:
        g = field.shape[0]
        xs = -EXTENT + (np.arange(g) + 0.5) / g * 2 * EXTENT
        data.append(dict(type="heatmap", z=L(field[::-1]), x=L(xs), y=L(xs), colorscale=VIRIDIS_SCALE, zmin=0, zmax=1,
                         showscale=False, zsmooth="best", hovertemplate="p(class 1)=%{z:.2f}<extra></extra>"))
    for cls, fill, edge in ((0, "#ffffff", "#05070a"), (1, "#05070a", "#ffffff")):
        m = train.y == cls
        data.append(dict(type="scatter", mode="markers", x=L(train.x[m, 0]), y=L(train.x[m, 1]), hoverinfo="skip",
                         marker=dict(size=6, color=fill, line=dict(color=edge, width=1))))
        if show_test:
            t = test.y == cls
            data.append(dict(type="scatter", mode="markers", x=L(test.x[t, 0]), y=L(test.x[t, 1]), hoverinfo="skip",
                             marker=dict(size=9, symbol="circle-open", color=edge if cls == 0 else "#ffffff", line=dict(color=edge if cls == 0 else "#ffffff", width=2))))
    return _fig(data, _layout(dark, height, margin=dict(l=24, r=8, t=8, b=24), showlegend=False, **_square_axes(dark, EXTENT)))


def neuron_figure(grid: np.ndarray, unit: int, dark: bool = True, height: int = 170) -> Figure:
    """One neuron's activation over the input plane. grid: (g, g, nout)."""
    z = grid[:, :, unit]
    span = max(float(z.max() - z.min()), 1e-9)
    data = [dict(type="heatmap", z=L((z[::-1] - z.min()) / span), colorscale=VIRIDIS_SCALE, showscale=False, zsmooth="best",
                 customdata=L(z[::-1]), hovertemplate="activation %{customdata:.3f}<extra></extra>")]
    return _fig(data, _layout(dark, height, margin=dict(l=4, r=4, t=4, b=4), showlegend=False,
                              xaxis=dict(visible=False, constrain="domain"),
                              yaxis=dict(visible=False, scaleanchor="x", constrain="domain")))


def network_layout(specs: list[LayerSpec], width: float = 100.0, height: float = 100.0):
    """Node coordinates per column (input, hidden..., output)."""
    cols = [2] + [s.nout for s in specs]
    pos = []
    for c, n in enumerate(cols):
        x = 8 + c * (width - 16) / max(1, len(cols) - 1)
        ys = [height / 2] if n == 1 else [8 + (u + 0.5) * (height - 16) / n for u in range(n)]
        pos.append([(x, y) for y in ys])
    return pos


def network_figure(specs: list[LayerSpec], snap: Snapshot | None, selected: tuple[int, int] | None, dark: bool = True,
                   height: int = 330) -> Figure:
    """Neurons as nodes, weights as edges: red = positive, blue = negative, thicker = larger."""
    pos = network_layout(specs)
    data: list[dict] = []
    size = float(np.clip(260 / max(len(c) for c in pos), 9, 26))
    if snap is not None:
        max_abs = max(max(float(np.max(np.abs(W))) for W, _ in snap.weights), 1e-9)
        buckets: dict[tuple[int, int], tuple[list, list]] = {}
        for l, (W, _) in enumerate(snap.weights):
            bucket = np.minimum(4, (5 * np.abs(W) / max_abs).astype(int))
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    xs, ys = buckets.setdefault((int(W[i, j] >= 0), int(bucket[i, j])), ([], []))
                    xs += [pos[l][i][0], pos[l + 1][j][0], None]
                    ys += [pos[l][i][1], pos[l + 1][j][1], None]
        for (positive, b), (xs, ys) in sorted(buckets.items(), key=lambda kv: kv[0][1]):
            data.append(dict(type="scatter", mode="lines", x=xs, y=ys, hoverinfo="skip", opacity=0.18 + 0.16 * b,
                             line=dict(color=RED if positive else BLUE, width=0.5 + 1.1 * (b + 1))))
    nx, ny, color, text, custom, symbol, lw, lc = [], [], [], [], [], [], [], []
    for c, col in enumerate(pos):
        for u, (x, y) in enumerate(col):
            layer = c - 1
            nx.append(x)
            ny.append(y)
            custom.append([layer, u])
            if layer < 0 or snap is None:
                color.append("#6b7686")
                text.append(("input x" if u == 0 else "input y") if layer < 0 else "")
                symbol.append("circle")
            else:
                st = snap.layers[layer]
                m, act = st.unit_mean[u], specs[layer].activation
                t = (m + 1) / 2 if act == "tanh" else m if act == "sigmoid" else m / max(0.5, max(st.unit_mean))
                color.append(rgb_hex(viridis(np.array(float(np.clip(t, 0, 1))))))
                symbol.append("x" if act in ("relu", "gelu") and m < 1e-3 and st.dead_pct > 0 else "circle")
                text.append(f"layer {layer + 1} · unit {u + 1}<br>mean activation {m:.3f}" if layer < len(specs) - 1
                            else f"output ŷ<br>mean P(class 1) {m:.3f}")
            sel = selected is not None and selected == (layer, u)
            lw.append(3 if sel else 1)
            lc.append(TEAL if sel else ("#243040" if dark else "#9aa7b8"))
    data.append(dict(type="scatter", mode="markers", x=nx, y=ny, customdata=custom, text=text, hoverinfo="text",
                     marker=dict(size=size, color=color, symbol=symbol, line=dict(width=lw, color=lc))))
    names = ["input"] + [f"hidden {i + 1}" for i in range(len(specs) - 1)] + ["output"]
    annotations = [dict(x=pos[c][0][0], y=98, text=name, showarrow=False, font=dict(size=10)) for c, name in enumerate(names)]
    return _fig(data, _layout(dark, height, margin=dict(l=4, r=4, t=4, b=4), showlegend=False, clickmode="event",
                              xaxis=dict(visible=False, range=[0, 100]), yaxis=dict(visible=False, range=[0, 100]),
                              annotations=annotations))


def landscape_figure(land: Landscape, path: list[tuple[float, float]], dark: bool = True, height: int = 300) -> Figure:
    r = land.range_
    axis = np.linspace(-r, r, land.grid)
    data: list[dict] = [dict(type="heatmap", z=L(land.values[::-1]), x=L(axis), y=L(axis), colorscale=VIRIDIS_SCALE,
                             zmin=land.min, zmax=max(land.clip, land.min + 1e-9), zsmooth="best",
                             colorbar=dict(title="loss", thickness=8))]
    if path:
        px, py = zip(*path)
        data.append(dict(type="scatter", mode="lines", x=list(px), y=list(py), hoverinfo="skip", line=dict(color="#ffffff", width=1.6)))
        data.append(dict(type="scatter", mode="markers", x=[px[0]], y=[py[0]], hoverinfo="skip",
                         marker=dict(size=8, color="#05070a", line=dict(color="#ffffff", width=1.5))))
    data.append(dict(type="scatter", mode="markers", x=[0], y=[0], hovertemplate="current weights<extra></extra>",
                     marker=dict(size=11, color=TEAL, line=dict(color="#05070a", width=1.5))))
    return _fig(data, _layout(dark, height, margin=dict(l=30, r=8, t=8, b=28), showlegend=False,
                              xaxis=dict(range=[-r, r], constrain="domain", title="direction 1"),
                              yaxis=dict(range=[-r, r], scaleanchor="x", constrain="domain", title="direction 2")))


def compare_figure(a: Series, b: Series, kind: str, dark: bool = True, height: int = 210) -> Figure:
    """Overlay run A (teal) and run B (violet); dashed = test."""
    ya = (a.train_loss, a.test_loss) if kind == "loss" else (a.train_acc, a.test_acc)
    yb = (b.train_loss, b.test_loss) if kind == "loss" else (b.train_acc, b.test_acc)
    yaxis = dict(rangemode="tozero") if kind == "loss" else dict(range=[0, 1.02], tickformat=".0%")
    return _fig([line(a.steps, ya[0], "A train", TEAL), line(b.steps, yb[0], "B train", VIOLET),
                 line(a.steps, ya[1], "A test", TEAL, 1.6, "dash"), line(b.steps, yb[1], "B test", VIOLET, 1.6, "dash")],
                _layout(dark, height, yaxis=yaxis))


# ------------------------------------------------------------------------------------ MNIST
def confusion_figure(conf: np.ndarray, dark: bool = True, height: int = 320) -> Figure:
    rows = conf.sum(axis=1, keepdims=True)
    frac = np.divide(conf, rows, out=np.zeros(conf.shape, dtype=float), where=rows > 0)
    annotations = [
        dict(x=p, y=t, text=str(int(conf[t, p])), showarrow=False, font=dict(size=10, color="#05070a" if frac[t, p] > 0.5 else "#ffffff"))
        for t in range(10) for p in range(10) if conf[t, p]
    ]
    data = [dict(type="heatmap", z=L(np.sqrt(frac)), colorscale=VIRIDIS_SCALE, zmin=0, zmax=1, showscale=False, customdata=conf.tolist(),
                 hovertemplate="true %{y} → predicted %{x}: %{customdata}<extra></extra>")]
    return _fig(data, _layout(dark, height, margin=dict(l=36, r=8, t=8, b=36), showlegend=False, annotations=annotations,
                              xaxis=dict(title="predicted", dtick=1, constrain="domain"),
                              yaxis=dict(title="true", dtick=1, autorange="reversed", scaleanchor="x", constrain="domain")))


def mosaic(maps: np.ndarray, cols: int, gap: int = 1) -> np.ndarray:
    """Tile (k, h, w) images into one array (NaN gaps)."""
    k, h, w = maps.shape
    rows = -(-k // cols)
    out = np.full((rows * (h + gap) - gap, cols * (w + gap) - gap), np.nan)
    for i in range(k):
        r, c = divmod(i, cols)
        out[r * (h + gap) : r * (h + gap) + h, c * (w + gap) : c * (w + gap) + w] = maps[i]
    return out


def mosaic_figure(maps: np.ndarray, cols: int, signed: bool, dark: bool = True, height: int = 220) -> Figure:
    """Filters (signed, blue/red) or feature maps (positive, viridis), each tile normalised separately."""
    norm = np.empty(maps.shape, dtype=np.float64)
    for i, m in enumerate(maps):
        if signed:
            norm[i] = m / (np.abs(m).max() or 1.0)
        else:
            lo, hi = float(m.min()), float(m.max())
            norm[i] = (m - lo) / ((hi - lo) or 1.0)
    data = [dict(type="heatmap", z=L(mosaic(norm, cols)), colorscale="RdBu" if signed else VIRIDIS_SCALE, reversescale=signed,
                 zmin=-1 if signed else 0, zmax=1, showscale=False, hoverinfo="skip")]
    return _fig(data, _layout(dark, height, margin=dict(l=2, r=2, t=2, b=2), showlegend=False,
                              xaxis=dict(visible=False, constrain="domain"),
                              yaxis=dict(visible=False, autorange="reversed", scaleanchor="x", constrain="domain")))


def mnist_curves_figure(points, kind: str, dark: bool = True, height: int = 190) -> Figure:
    ev = [p for p in points if p.test_loss is not None]
    if kind == "loss":
        tr, te, yaxis = [p.train_loss for p in points], [p.test_loss for p in ev], dict(rangemode="tozero")
    else:
        tr, te, yaxis = [p.train_acc for p in points], [p.test_acc for p in ev], dict(range=[0, 1.02], tickformat=".0%")
    return _fig([line([p.step for p in points], tr, "train", TEAL), line([p.step for p in ev], te, "test", AMBER)],
                _layout(dark, height, yaxis=yaxis))
