"""Landing page: a live, self-training network in the hero, then what the app can do."""

from __future__ import annotations

from nicegui import ui

from ..config import DEFAULT_CONFIG
from ..forward import forward_grid
from ..runner import Run
from . import charts
from .components import frame

FEATURES = [
    ("Playground", "/playground", "Pick a dataset, build a network, press Play and watch the decision boundary, every weight, gradient and activation update live. Drag the timeline to rewind."),
    ("Model Doctor", "/playground", "Deterministic rules read the run's telemetry and diagnose dead ReLUs, vanishing gradients, a learning rate that is too high, overfitting and more, with a one-click fix."),
    ("Compare", "/compare", "Two networks on identical data and seed with one setting different: Adam vs SGD, ReLU vs tanh, wide vs deep. The loss curves tell the story."),
    ("Backprop, step by step", "/backprop", "One full training step of a 2-3-1 network with real numbers: forward, loss, backward, update. Every gradient is checked against a hand-written chain rule."),
    ("Real digits (MNIST)", "/mnist", "A convolutional network written from scratch in NumPy learns handwritten digits on your CPU. Inspect its filters and feature maps, then draw your own digit."),
    ("Share an experiment", "/playground", "One click copies a link that reopens the exact same dataset, architecture and hyper-parameters."),
]


def build() -> None:
    dark = frame("Home")
    run = Run(DEFAULT_CONFIG, speed=4)
    ui.context.client.on_disconnect(run.close)
    st = {"dark": dark.value}

    with ui.column().classes("w-full max-w-[1200px] mx-auto p-4 gap-8"):
        with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-2 gap-6 items-center pt-4"):
            with ui.column().classes("gap-4"):
                ui.label("Watch a neural network learn.").classes("text-4xl font-bold leading-tight")
                ui.label("LearnScope trains real networks in front of you: the decision boundary bends, the weights move, the loss falls. "
                         "Pause, rewind to any moment, ask the Model Doctor what went wrong. The whole engine, including backprop and a "
                         "convolutional network, is written from scratch in Python and NumPy.").classes("text-base ls-muted")
                with ui.row().classes("gap-2"):
                    ui.button("Open the Playground", icon="play_arrow", on_click=lambda: ui.navigate.to("/playground?play=1")).props("color=primary no-caps size=lg")
                    ui.button("How backprop works", icon="school", on_click=lambda: ui.navigate.to("/backprop")).props("outline no-caps size=lg")
                stats = ui.label().classes("num text-xs ls-muted")
            with ui.card().classes("ls-card p-2 no-shadow w-full"):
                hero = ui.plotly(charts.boundary_figure(None, run.train, run.test, st["dark"], 380)).classes("w-full")
                ui.label("A live run: a spiral, learned by a 2×8 tanh network with Adam. Nothing here is a recording.").classes("text-[11px] ls-muted")

        with ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3"):
            for title, href, text in FEATURES:
                with ui.link(target=href).classes("no-underline text-inherit"):
                    with ui.column().classes("ls-card p-4 gap-1 h-full hover:border-teal-400"):
                        ui.label(title).classes("text-sm font-bold text-teal-400")
                        ui.label(text).classes("text-sm ls-muted")

        ui.label("Python · NumPy · NiceGUI · Plotly. No PyTorch, no TensorFlow: every gradient is derived by hand and verified numerically.").classes("text-xs ls-muted text-center w-full pb-6")

    def refresh() -> None:
        if dark.value != st["dark"]:
            st["dark"] = dark.value
        snap = run.view_snapshot()
        if snap is None:
            return
        hero.update_figure(charts.boundary_figure(forward_grid(run.specs, snap.weights)[-1][:, :, 0], run.train, run.test, st["dark"], 380))
        stats.set_text(f"live: step {snap.step} · test accuracy {snap.test_acc * 100:.0f}% · loss {snap.test_loss:.3f}")
        if snap.step >= 2500 and run.running:  # converged: stop burning CPU
            run.pause()

    ui.timer(0.4, refresh)
    run.start()
