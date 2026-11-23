"""Reusable UI pieces: page frame, panels with "What am I looking at?" notes, settings editor, doctor cards."""

from __future__ import annotations

import math
from collections.abc import Callable

from nicegui import ui

from ..config import (
    ACTIVATION_LABELS,
    DATASET_LABELS,
    OPTIMIZER_LABELS,
    HiddenLayer,
)
from ..doctor import Diagnosis
from ..runner import Run

NAV = [("Home", "/"), ("Playground", "/playground?play=0"), ("Compare", "/compare"), ("Backprop", "/backprop"),
       ("MNIST", "/mnist"), ("About", "/about")]

CSS = """
.ls-card { border: 1px solid var(--ls-line); border-radius: 12px; background: var(--ls-panel); }
.body--dark { --ls-line:#243040; --ls-panel:#0e1218; --ls-muted:#8b98a8; }
.body--light { --ls-line:#d5dce5; --ls-panel:#ffffff; --ls-muted:#5a6778; }
.ls-muted { color: var(--ls-muted); }
.ls-flash { box-shadow: 0 0 0 2px #2dd4bf; }
.num { font-variant-numeric: tabular-nums; font-family: ui-monospace, Consolas, monospace; }
.q-slider { padding: 0 6px; }
"""


def frame(title: str = "LearnScope", dark_default: bool = True) -> ui.dark_mode:
    """Page chrome: colours, header with navigation and the theme toggle. Returns the dark-mode switch."""
    ui.colors(primary="#2dd4bf", accent="#2dd4bf")
    ui.add_css(CSS)
    dark = ui.dark_mode(dark_default)
    with ui.header(elevated=False).classes("items-center gap-4 px-4 py-2 border-b").style("background: transparent; color: inherit; border-color: var(--ls-line)"):
        with ui.link(target="/").classes("no-underline text-inherit"):
            ui.html('<span style="font-weight:700;font-size:1.05rem">Learn<span style="color:#2dd4bf">Scope</span></span>', sanitize=False)
        with ui.row().classes("gap-1"):
            for label, href in NAV:
                ui.link(label, href).classes("px-2 py-1 rounded no-underline text-sm ls-muted hover:text-teal-400")
        ui.space()
        ui.label("Space play/pause · ← → step · R reset · L live").classes("text-xs ls-muted gt-sm")
        ui.button(icon="contrast", on_click=dark.toggle).props("flat round dense").tooltip("Toggle light / dark theme")
    return dark


def info_button(text: str, title: str = "What am I looking at?") -> None:
    """ⓘ button that opens a short plain-English explanation."""
    with ui.button(icon="info").props("flat round dense size=sm").classes("ls-muted").props(f'aria-label="{title}"'):
        with ui.menu().props("anchor='bottom right' self='top right'"):
            with ui.card().classes("max-w-xs"):
                ui.label(title).classes("text-sm font-bold text-teal-400")
                ui.label(text).classes("text-xs")


class Panel:
    """A titled card. Use as a context manager; `.card` is the element (for flash highlighting)."""

    def __init__(self, title: str, info: str | None = None, classes: str = "", actions: Callable[[], None] | None = None) -> None:
        self.card = ui.card().classes(f"ls-card w-full p-3 gap-1 no-shadow {classes}")
        with self.card:
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                ui.label(title.upper()).classes("text-xs font-bold tracking-wider ls-muted")
                with ui.row().classes("items-center gap-1 no-wrap"):
                    if actions:
                        actions()
                    if info:
                        info_button(info)

    def __enter__(self):
        self.card.__enter__()
        return self

    def __exit__(self, *exc):
        return self.card.__exit__(*exc)

    def flash(self, seconds: float = 1.8) -> None:
        self.card.classes("ls-flash")
        ui.timer(seconds, lambda: self.card.classes(remove="ls-flash"), once=True)


def stat_tile(label: str, color: str = "") -> ui.label:
    with ui.column().classes("ls-card px-3 py-1 gap-0"):
        ui.label(label.upper()).classes("text-[10px] tracking-wider ls-muted")
        return ui.label("–").classes(f"num text-sm font-bold {color}")


def lr_to_slider(lr: float) -> float:
    return math.log10(lr)


def slider_to_lr(v: float) -> float:
    return float(f"{10 ** v:.2g}")


def fmt_lr(lr: float) -> str:
    return f"{lr:.2f}" if lr >= 0.1 else f"{lr:.3f}" if lr >= 0.01 else f"{lr:.2g}"


def slider_row(label: str, value: float, lo: float, hi: float, step: float, on_release: Callable[[float], None],
               fmt: Callable[[float], str] = lambda v: f"{v:g}", tip: str | None = None) -> ui.slider:
    with ui.column().classes("w-full gap-0"):
        with ui.row().classes("w-full justify-between items-baseline no-wrap"):
            lab = ui.label(label).classes("text-xs ls-muted")
            if tip:
                lab.tooltip(tip)
            shown = ui.label(fmt(value)).classes("num text-xs")
        s = ui.slider(min=lo, max=hi, step=step, value=value).props("dense")
        s.on_value_change(lambda e: shown.set_text(fmt(e.value)))
        s.on("change", lambda: on_release(s.value))  # fires when the user lets go: one rebuild per drag, not hundreds
    return s


def settings_panel(run: Run, *, data: bool = True, on_change: Callable[[], None] | None = None, shared_data_setter: Callable[..., None] | None = None) -> None:
    """Data / model / training controls bound to a Run. Learning rate, L2 and batch size apply live;
    everything else restarts the run (the data and architecture change)."""
    cfg = run.config

    def apply(**patch) -> None:
        run.set_config(**patch)
        if on_change:
            on_change()

    def apply_data(**patch) -> None:
        (shared_data_setter or apply)(**patch)
        if on_change:
            on_change()

    if data:
        with Panel("Data", "The 2-D toy datasets the network learns to separate. Each point is class 0 (white) or class 1 (black). "
                           "Noise blurs the classes. Solid dots train the network; rings are the held-out test set that checks it generalises."):
            ui.select({k: v for k, v in DATASET_LABELS.items()}, value=cfg.dataset, label="Dataset",
                      on_change=lambda e: apply_data(dataset=e.value)).classes("w-full")
            slider_row("Noise", cfg.noise, 0, 0.5, 0.01, lambda v: apply_data(noise=v), lambda v: f"{v:.2f}")
            slider_row("Samples", cfg.samples, 100, 1000, 50, lambda v: apply_data(samples=int(v)), lambda v: f"{int(v)}")
            slider_row("Test split", cfg.test_split, 0.1, 0.5, 0.05, lambda v: apply_data(test_split=v), lambda v: f"{round(v * 100)}%")
            ui.button("New random seed", icon="casino", on_click=lambda: apply_data(seed=int(__import__("random").random() * 99999))).props("flat dense").classes("w-full")

    with Panel("Model", "The network is a stack of dense layers. Each hidden layer has some neurons and an activation function that bends straight "
                        "lines into curves. The last layer is always one sigmoid neuron: the probability of class 1. Changing the architecture restarts training."):
        @ui.refreshable
        def layers_editor() -> None:
            layers = list(run.config.layers)
            for i, l in enumerate(layers):
                with ui.column().classes("w-full ls-card p-2 gap-1"):
                    with ui.row().classes("w-full justify-between items-center no-wrap"):
                        ui.label(f"Hidden layer {i + 1}").classes("text-xs font-bold")
                        ui.button(icon="close", on_click=lambda i=i: (apply(layers=tuple(x for k, x in enumerate(run.config.layers) if k != i)), layers_editor.refresh())
                                  ).props("flat round dense size=sm").set_enabled(len(layers) > 1)
                    slider_row("Neurons", l.units, 1, 16, 1, lambda v, i=i: _set_layer(i, units=int(v)), lambda v: f"{int(v)}")
                    ui.select(ACTIVATION_LABELS, value=l.activation, label="Activation",
                              on_change=lambda e, i=i: _set_layer(i, activation=e.value)).classes("w-full")
            ui.button("Add hidden layer", icon="add", on_click=lambda: (
                apply(layers=(*run.config.layers, HiddenLayer(6, run.config.layers[-1].activation))), layers_editor.refresh()
            )).props("flat dense").classes("w-full").set_enabled(len(layers) < 5)

        def _set_layer(i: int, **change) -> None:
            ls = list(run.config.layers)
            ls[i] = HiddenLayer(change.get("units", ls[i].units), change.get("activation", ls[i].activation))
            apply(layers=tuple(ls))

        layers_editor()

    with Panel("Training", "How the network learns. The optimizer decides how to use each gradient, the learning rate is the step size and the batch "
                           "size is how many points each step uses. L2 gently pulls weights toward zero to fight overfitting. Learning rate, L2 and "
                           "batch size apply instantly; the rest restarts training."):
        ui.select(OPTIMIZER_LABELS, value=cfg.optimizer, label="Optimizer", on_change=lambda e: apply(optimizer=e.value)).classes("w-full")
        slider_row("Learning rate", lr_to_slider(cfg.lr), -3.5, 0.47, 0.02, lambda v: apply(lr=slider_to_lr(v)),
                   lambda v: fmt_lr(slider_to_lr(v)), "Step size. Too small: slow. Too large: unstable.")
        slider_row("Batch size", cfg.batch_size, 1, 256, 1, lambda v: apply(batch_size=int(v)), lambda v: f"{int(v)}")
        slider_row("L2 regularization", cfg.l2, 0, 0.05, 0.001, lambda v: apply(l2=v), lambda v: f"{v:.3f}")


SEVERITY_COLOR = {"critical": "negative", "warn": "warning", "info": "positive"}
SEVERITY_ICON = {"critical": "error", "warn": "warning", "info": "check_circle"}


def doctor_cards(run: Run, diagnoses: list[Diagnosis], on_show: Callable[[Diagnosis], None] | None = None) -> None:
    """Render the Model Doctor's findings. Call inside a refreshable container."""
    if not diagnoses:
        ui.label("Watching the run… diagnoses appear after a few dozen steps.").classes("text-xs ls-muted")
        return
    for d in diagnoses:
        border = {"critical": "border-red-500", "warn": "border-amber-400", "info": ""}[d.severity]
        with ui.column().classes(f"w-full ls-card p-2 gap-1 border {border}").props(f'aria-label="{d.severity} diagnosis: {d.title}"'):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon(SEVERITY_ICON[d.severity], color=SEVERITY_COLOR[d.severity]).classes("text-base")
                    ui.label(d.title).classes("text-sm font-bold")
                ui.badge(d.severity, color=SEVERITY_COLOR[d.severity]).props("outline")
            ui.label(d.explanation).classes("text-xs ls-muted")
            ui.label(f"Evidence: step {d.evidence.step} — {d.evidence.detail}").classes("text-[11px] ls-muted")
            with ui.row().classes("gap-1"):
                ui.button("Show me", icon="search", on_click=lambda d=d: (run.jump_to_step(d.evidence.step), on_show and on_show(d))).props("flat dense size=sm")
                if d.fix is not None:
                    ui.button(d.fix.label, icon="auto_fix_high", on_click=lambda d=d: run.apply_fix(d.fix)).props("dense size=sm color=primary no-caps")


def diagnosis_signature(diagnoses: list[Diagnosis]) -> tuple:
    return tuple((d.id, d.severity, d.evidence.step, d.explanation) for d in diagnoses)
