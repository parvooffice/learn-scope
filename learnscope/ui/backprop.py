"""The Backprop explainer: one full training step on a 2-3-1 network, with real numbers.

Every value shown comes from the autograd engine (learnscope/autograd.py) and is cross-checked
against the chain rule written out by hand."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

from nicegui import ui

from ..explainer import ExplainerParams, apply_step, initial_params, run_explainer
from . import charts
from .components import Panel, frame, slider_row

STEP_COUNT = 10
SUB = "₀₁₂₃₄₅₆₇₈₉"
NODE = {"x1": (70, 105), "x2": (70, 255), "h": [(290, 70), (290, 180), (290, 295)], "y": (500, 180), "L": (640, 180)}


def f(n: float, d: int = 4) -> str:
    return f"{(0.0 if n == 0 else n):.{d}f}"


def sgn(n: float) -> str:
    return f"({f(n)})" if n < 0 else f(n)


@dataclass
class StepInfo:
    phase: str
    title: str
    idea: str
    lines: list[str]


def build_steps(p: ExplainerParams, r, lr: float, nxt: ExplainerParams, r_next) -> list[StepInfo]:
    x, t, w, b, v, c = p.x, p.t, p.w, p.b, p.v, p.c
    s = lambda j: SUB[j + 1]  # noqa: E731
    return [
        StepInfo("setup", "0 · The network and its numbers",
                 "A neural network is just a formula with adjustable numbers (weights w, biases b, v, c). Training means nudging those numbers so the output ŷ gets close to "
                 "the target t. Below, everything is computed live by our own autograd engine.",
                 [f"inputs: x₁ = {f(x[0], 2)}, x₂ = {f(x[1], 2)}      target: t = {f(t, 2)}",
                  "hidden layer (3 neurons, tanh):  z = w·x + b,   h = tanh(z)",
                  "output (linear):  ŷ = v₁h₁ + v₂h₂ + v₃h₃ + c",
                  "loss:  L = (ŷ − t)²   (how wrong we are; we want it near 0)"]),
        StepInfo("forward", "1 · Forward pass: weighted sums",
                 "Each hidden neuron multiplies every input by a weight, adds them up, and adds its bias. This is the weighted sum z.",
                 [f"z{s(j)} = w{s(j)}₁·x₁ + w{s(j)}₂·x₂ + b{s(j)} = {sgn(w[j][0])}({f(x[0], 2)}) + {sgn(w[j][1])}({f(x[1], 2)}) + {sgn(b[j])} = {f(z)}"
                  for j, z in enumerate(r.z)]),
        StepInfo("forward", "2 · Forward pass: the activation",
                 "tanh squashes z into the range −1…1. Without a bend like this, stacking layers would still be just one straight line.",
                 [f"h{s(j)} = tanh({f(r.z[j])}) = {f(h)}" for j, h in enumerate(r.h)]),
        StepInfo("forward", "3 · Forward pass: the output",
                 "The output neuron takes a weighted sum of the hidden values. That is our prediction ŷ.",
                 [f"ŷ = {' + '.join(f'{sgn(vj)}({f(r.h[j])})' for j, vj in enumerate(v))} + {sgn(c)}", f"ŷ = {f(r.yhat)}"]),
        StepInfo("loss", "4 · The loss",
                 "We measure the mistake with the squared error. Squaring makes it always positive and punishes big mistakes more. Learning = making this number smaller.",
                 [f"L = (ŷ − t)² = ({f(r.yhat)} − {f(t, 2)})² = {f(r.loss)}"]),
        StepInfo("backward", "5 · Backward pass: start at the loss",
                 "Backpropagation asks: \"if I nudge ŷ a tiny bit, how much does L change?\" That sensitivity is the derivative ∂L/∂ŷ. It is the first link of the chain; "
                 "every other gradient is built from it.",
                 [f"∂L/∂ŷ = 2(ŷ − t) = 2({f(r.yhat)} − {f(t, 2)}) = {f(r.manual.yhat)}", f"autograd says: {f(r.auto.yhat)}"]),
        StepInfo("backward", "6 · Backward: through the output neuron",
                 "Chain rule: (how L changes with ŷ) × (how ŷ changes with the thing). Since ŷ = Σ v·h + c, the \"how ŷ changes with v\" is just h, and with h it is just v.",
                 [*[f"∂L/∂v{s(j)} = ∂L/∂ŷ · h{s(j)} = {f(r.manual.yhat)} · {f(h)} = {f(r.manual.v[j])}" for j, h in enumerate(r.h)],
                  f"∂L/∂c = ∂L/∂ŷ · 1 = {f(r.manual.c)}",
                  *[f"∂L/∂h{s(j)} = ∂L/∂ŷ · v{s(j)} = {f(r.manual.yhat)} · {sgn(vj)} = {f(r.manual.h[j])}" for j, vj in enumerate(v)]]),
        StepInfo("backward", "7 · Backward: through tanh",
                 "The slope of tanh at z is 1 − tanh(z)² = 1 − h². Where tanh is flat (h near ±1) the slope is near 0 and the gradient gets squashed. That is why deep "
                 "tanh/sigmoid networks suffer vanishing gradients.",
                 [f"∂L/∂z{s(j)} = ∂L/∂h{s(j)} · (1 − h{s(j)}²) = {f(r.manual.h[j])} · {f(1 - h * h)} = {f(r.manual.z[j])}" for j, h in enumerate(r.h)]),
        StepInfo("backward", "8 · Backward: to the first-layer weights",
                 "Since z = w₁x₁ + w₂x₂ + b, the slope of z with respect to w₁ is x₁, and with respect to b is 1. Now every weight in the network knows which way (and how "
                 "strongly) it should move.",
                 [*[f"∂L/∂w{s(j)}₁ = {f(r.manual.z[j])} · {f(x[0], 2)} = {f(r.manual.w[j][0])}    ∂L/∂w{s(j)}₂ = {f(r.manual.z[j])} · {f(x[1], 2)} = {f(r.manual.w[j][1])}    "
                     f"∂L/∂b{s(j)} = {f(r.manual.b[j])}" for j in range(3)],
                  f"✓ hand-written chain rule vs autograd engine: max difference = {r.max_diff:.1e}"]),
        StepInfo("update", "9 · The update: gradient descent",
                 f"Move every parameter a small step against its gradient: p ← p − lr · ∂L/∂p (lr = {lr}). Then the loss goes down. Repeat thousands of times: that is training.",
                 [f"w₁₁: {f(w[0][0])} → {f(nxt.w[0][0])}     v₁: {f(v[0])} → {f(nxt.v[0])}     b₁: {f(b[0])} → {f(nxt.b[0])}   (… all 16 parameters)",
                  f"loss before = {f(r.loss, 5)}   →   loss after = {f(r_next.loss, 5)}   {'✓ lower' if r_next.loss < r.loss else '⚠ higher: lr too big'}"]),
    ]


def diagram(p: ExplainerParams, r, step: int) -> str:
    """The network as SVG. Values appear as the computation reaches them; amber ∂ values are gradients."""
    show_z, show_h, show_y, show_l = step >= 1, step >= 2, step >= 3, step >= 4
    g1, g2, g3, g4 = step >= 5, step >= 6, step >= 7, step >= 8
    out = ['<svg viewBox="0 0 710 410" style="width:100%;max-height:520px" role="img" aria-label="Diagram of the 2-3-1 network with values and gradients for the current step">']

    def edge(x1, y1, x2, y2, wv, hi, grad=None, tpos=0.5):
        sw = 0.8 + min(4, abs(wv) * 3)
        color = "#f0625c" if wv >= 0 else "#4f8cff"
        mx, my = x1 + (x2 - x1) * tpos, y1 + (y2 - y1) * tpos
        out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw + (2 if hi else 0):.2f}" opacity="{1 if hi else 0.5}"/>')
        out.append(f'<text x="{mx}" y="{my - 3}" text-anchor="middle" font-size="10" class="num" fill="currentColor" stroke="var(--ls-panel)" stroke-width="3" paint-order="stroke">{wv:.2f}</text>')
        if grad is not None:
            out.append(f'<text x="{mx}" y="{my + 10}" text-anchor="middle" font-size="10" class="num" fill="#f5b942" stroke="var(--ls-panel)" stroke-width="3" paint-order="stroke">∂ {grad:.3f}</text>')

    def node(cx, cy, top, lines, hi, rad=24):
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{rad}" fill="none" stroke="{"#2dd4bf" if hi else "currentColor"}" stroke-opacity="{1 if hi else 0.35}" stroke-width="{3 if hi else 1.5}"/>')
        out.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">{top}</text>')
        for i, (txt, color) in enumerate(lines):
            out.append(f'<text x="{cx}" y="{cy + rad + 14 + i * 12}" text-anchor="middle" font-size="10" class="num" fill="{color}">{txt}</text>')

    MUTED, TEXT, AMBER = "#8b98a8", "currentColor", "#f5b942"
    for j, (hx, hy) in enumerate(NODE["h"]):
        edge(NODE["x1"][0] + 24, NODE["x1"][1], hx - 24, hy, p.w[j][0], step in (1, 8), r.manual.w[j][0] if g4 else None, 0.3)
        edge(NODE["x2"][0] + 24, NODE["x2"][1], hx - 24, hy, p.w[j][1], step in (1, 8), r.manual.w[j][1] if g4 else None, 0.3)
    for j, (hx, hy) in enumerate(NODE["h"]):
        edge(hx + 24, hy, NODE["y"][0] - 24, NODE["y"][1], p.v[j], step in (3, 6), r.manual.v[j] if g2 else None, 0.5)
    out.append(f'<line x1="{NODE["y"][0] + 24}" y1="180" x2="{NODE["L"][0] - 24}" y2="180" stroke="{MUTED}" stroke-width="{4 if step in (4, 5) else 1.5}" opacity="0.7"/>')

    node(*NODE["x1"], "x₁", [(f"{p.x[0]:.2f}", TEXT)], False)
    node(*NODE["x2"], "x₂", [(f"{p.x[1]:.2f}", TEXT)], False)
    for j, (hx, hy) in enumerate(NODE["h"]):
        lines = [(f"z = {r.z[j]:.3f}", MUTED)] if show_z else [(f"b = {p.b[j]:.2f}", MUTED)]
        if show_h:
            lines.append((f"h = {r.h[j]:.3f}", TEXT))
        if g2 and not g3:
            lines.append((f"∂h {r.manual.h[j]:.3f}", AMBER))
        if g3:
            lines.append((f"∂z {r.manual.z[j]:.3f}", AMBER))
        if g4:
            lines.append((f"∂b {r.manual.b[j]:.3f}", AMBER))
        node(hx, hy, f"h{SUB[j + 1]}", lines, step in (2, 7))
    ylines = [(f"ŷ = {r.yhat:.3f}", TEXT)] if show_y else [(f"c = {p.c:.2f}", MUTED)]
    if g1:
        ylines.append((f"∂ŷ {r.manual.yhat:.3f}", AMBER))
    if g2:
        ylines.append((f"∂c {r.manual.c:.3f}", AMBER))
    node(*NODE["y"], "ŷ", ylines, step in (3, 5, 6))
    node(*NODE["L"], "L", [(f"t = {p.t:.2f}", MUTED), *([(f"L = {r.loss:.4f}", TEXT)] if show_l else [])], step == 4, 22)
    phase = "gradients flow backward ⟵" if 5 <= step <= 8 else "values flow forward ⟶" if 1 <= step <= 3 else ""
    out.append(f'<text x="355" y="16" text-anchor="middle" font-size="11" fill="{MUTED}">{phase}</text></svg>')
    return "".join(out)


def build() -> None:
    dark = frame("Backprop")
    st = SimpleNamespace(params=initial_params(4), lr=0.2, step=0, auto=False, losses=[])

    def compute():
        r = run_explainer(st.params)
        nxt = apply_step(st.params, r.auto, st.lr)
        return r, nxt, run_explainer(nxt)

    @ui.refreshable
    def content() -> None:
        r, nxt, rn = compute()
        steps = build_steps(st.params, r, st.lr, nxt, rn)
        cur = steps[st.step]
        with ui.element("div").classes("w-full grid grid-cols-1 xl:grid-cols-[1.3fr_1fr] gap-3"):
            with Panel("The network", "Each circle is a neuron and each line a weight (red positive, blue negative). Values appear as the computation reaches them. During the backward pass, "
                                      "amber numbers show the gradient ∂L/∂(thing): how much the loss would change if that thing changed a little."):
                ui.html(diagram(st.params, r, st.step), sanitize=False).classes("w-full")
                with ui.row().classes("items-center gap-2"):
                    ui.button("Back", icon="arrow_back", on_click=lambda: go(-1)).props("outline no-caps").set_enabled(st.step > 0)
                    ui.button("Next", icon="arrow_forward", on_click=lambda: go(1)).props("color=primary no-caps").set_enabled(st.step < STEP_COUNT - 1)
                    ui.button("Pause" if st.auto else "Auto-play", icon="pause" if st.auto else "play_arrow", on_click=toggle_auto).props("outline no-caps")
                    ui.label(f"step {st.step + 1} / {STEP_COUNT}").classes("num text-xs ls-muted")
                    ui.space()
                    ui.button("Apply update", icon="sync", on_click=apply_update).props("outline no-caps").tooltip("Apply the update and start the next iteration")
                    ui.button("Train ×100", on_click=lambda: train(100)).props("outline no-caps").tooltip("Run 100 full training steps instantly")
                    ui.button("Reset", on_click=reset).props("flat color=negative no-caps")
            with ui.column().classes("w-full gap-3"):
                with Panel(cur.title, "This panel explains the current step and shows the exact arithmetic with the live numbers from the diagram."):
                    ui.label(cur.idea).classes("text-sm")
                    ui.html("<pre class='num' style='white-space:pre-wrap;font-size:11.5px;margin:6px 0 0'>" + "\n".join(
                        l.replace("&", "&amp;").replace("<", "&lt;") for l in cur.lines) + "</pre>", sanitize=False).classes("ls-card p-2 w-full")
                with Panel("Play with it", "Change the inputs, the target or the learning rate; every number above is recomputed instantly by the autograd engine."):
                    with ui.element("div").classes("grid grid-cols-2 gap-x-4 gap-y-2 w-full"):
                        slider_row("x₁", st.params.x[0], -1.5, 1.5, 0.05, lambda v: setp(x=(v, st.params.x[1])), lambda v: f"{v:.2f}")
                        slider_row("x₂", st.params.x[1], -1.5, 1.5, 0.05, lambda v: setp(x=(st.params.x[0], v)), lambda v: f"{v:.2f}")
                        slider_row("target t", st.params.t, -1.5, 1.5, 0.05, lambda v: setp(t=v), lambda v: f"{v:.2f}")
                        slider_row("learning rate", st.lr, 0.01, 1.5, 0.01, lambda v: setlr(v), lambda v: f"{v:.2f}")
                with Panel("Loss over updates", "Each time you apply an update the loss is recorded. A well-chosen learning rate makes it fall smoothly toward zero; "
                                                 "a too-large learning rate makes it bounce or blow up (try 1.5)."):
                    if len(st.losses) > 1:
                        ui.plotly({"data": [charts.line(list(range(len(st.losses))), st.losses, "loss", charts.TEAL)],
                                   "layout": charts._layout(dark.value, 130, yaxis=dict(rangemode="tozero"), showlegend=False)}).classes("w-full")
                    else:
                        ui.label("Press “Apply update” or “Train ×100” to see the loss fall.").classes("text-xs ls-muted")

    def go(d: int) -> None:
        st.step = max(0, min(STEP_COUNT - 1, st.step + d))
        content.refresh()

    def setp(**kw) -> None:
        st.params = replace(st.params, **kw)
        content.refresh()

    def setlr(v: float) -> None:
        st.lr = v
        content.refresh()

    def apply_update() -> None:
        r, nxt, _ = compute()
        st.losses.append(r.loss)
        st.params, st.step = nxt, 0
        content.refresh()

    def train(n: int) -> None:
        p = st.params
        for _ in range(n):
            r = run_explainer(p)
            st.losses.append(r.loss)
            p = apply_step(p, r.auto, st.lr)
        st.params, st.step = p, 0
        content.refresh()

    def reset() -> None:
        st.params, st.losses, st.step, st.auto = initial_params(4), [], 0, False
        content.refresh()

    def toggle_auto() -> None:
        st.auto = not st.auto
        content.refresh()

    def tick() -> None:
        if st.auto:
            if st.step >= STEP_COUNT - 1:
                st.auto = False
            else:
                st.step += 1
            content.refresh()

    with ui.column().classes("w-full p-3 gap-3 max-w-[1300px] mx-auto"):
        ui.label("Backpropagation, step by step").classes("text-xl font-bold")
        ui.label("A 2-3-1 network with real numbers. Press Next to walk through one full training step: forward, loss, backward, update.").classes("text-sm ls-muted")
        content()
    ui.timer(1.7, tick)
    _ = time
