"""MNIST page: train a real digit classifier (CNN or MLP) in NumPy, watch it learn, draw your own digit."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from nicegui import ui

from ..mnist.data import load_mnist
from ..mnist.draw import CANVAS, stroke, to_mnist
from ..mnist.trainer import MODEL_LABELS, MnistConfig, MnistRun
from ..png import data_uri, upscale
from . import charts
from .components import Panel, fmt_lr, frame, lr_to_slider, slider_row, slider_to_lr, stat_tile

EMPTY = {"data": [], "layout": {}}
SPEEDS = {1: "1×", 4: "4×", 16: "16×"}


def digit_uri(img: np.ndarray, scale: int = 8) -> str:
    return data_uri(upscale((np.clip(img, 0, 1) * 255).astype(np.uint8), scale))


def probs_figure(probs: np.ndarray, truth: int | None, dark: bool) -> dict:
    pred = int(np.argmax(probs))
    colors = [charts.TEAL if i == pred else "#5a6778" for i in range(10)]
    layout = charts._layout(dark, 170, margin=dict(l=36, r=8, t=8, b=26), showlegend=False,
                            xaxis=dict(dtick=1, title="digit", type="category"), yaxis=dict(range=[0, 1.02], tickformat=".0%"))
    if truth is not None:
        layout["shapes"] = [dict(type="rect", xref="x", yref="paper", x0=truth - 0.5, x1=truth + 0.5, y0=0, y1=1,
                                 line=dict(color=charts.AMBER, width=1.5, dash="dot"))]
    return {"data": [dict(type="bar", x=[str(i) for i in range(10)], y=charts.L(probs), marker=dict(color=colors),
                          hovertemplate="digit %{x}: %{y:.1%}<extra></extra>")], "layout": layout}


def build() -> MnistRun:
    dark = frame("MNIST")
    run = MnistRun(MnistConfig(), load_mnist(), speed=4)
    ui.context.client.on_disconnect(run.close)
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    st = SimpleNamespace(version=-1, dark=dark.value, drawing=False, last=None, canvas_dirty=False, drawn=False, worst_sig=None)

    def mark_dirty() -> None:
        st.version = -1

    with ui.column().classes("w-full p-3 gap-3 max-w-[1400px] mx-auto"):
        ui.label("Real digits: a CNN built from scratch in NumPy").classes("text-xl font-bold")
        ui.label("Hand-written convolution, pooling and Adam (no PyTorch, no TensorFlow) learning to read handwritten digits from a 10,000-image MNIST subset. "
                 "Press Train, then draw your own digit.").classes("text-sm ls-muted")

        with ui.row().classes("items-center gap-2 w-full"):
            play = ui.button("Train", icon="play_arrow", on_click=lambda: (run.start() if not run.running else run.pause(), mark_dirty())).props("color=primary no-caps")
            ui.button("Step", on_click=lambda: (run.step(1), mark_dirty())).props("outline no-caps")
            ui.button("Reset", icon="restart_alt", on_click=lambda: (run.reset(), mark_dirty())).props("outline no-caps")
            ui.select(MODEL_LABELS, value=run.cfg.model, label="Model",
                      on_change=lambda e: (run.configure(model=e.value), mark_dirty())).classes("w-80").props("dense outlined")
            ui.label("speed").classes("text-xs ls-muted ml-2")
            ui.toggle(SPEEDS, value=4, on_change=lambda e: run.set_speed(e.value)).props("dense no-caps size=sm")
            ui.space()
            ui.badge("NumPy · CPU", color="grey").props("outline")
        error_label = ui.label().classes("text-red-400 text-sm")

        with ui.row().classes("gap-2"):
            t_step = stat_tile("step")
            t_epoch = stat_tile("epoch")
            t_tl = stat_tile("train loss", "text-teal-400")
            t_ta = stat_tile("train acc")
            t_va = stat_tile("test acc", "text-amber-400")
            t_ips = stat_tile("images / s")
            t_par = stat_tile("parameters")

        with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-3 gap-3 items-start"):
            # ------------------------------------------------------------ column 1: learning
            with ui.column().classes("w-full gap-3"):
                with Panel("Settings", "The learning rate is Adam's step size. A bigger batch gives smoother (but slower) updates. Changing the model restarts training."):
                    slider_row("Learning rate", lr_to_slider(run.cfg.lr), -3.5, -1.5, 0.02,
                               lambda v: (run.configure(lr=slider_to_lr(v)), mark_dirty()), lambda v: fmt_lr(slider_to_lr(v)))
                    slider_row("Batch size", run.cfg.batch_size, 8, 256, 8, lambda v: (run.configure(batch_size=int(v)), mark_dirty()), lambda v: f"{int(v)}")
                with Panel("Loss", "Cross-entropy: how surprised the network is by the right answer (lower is better). Teal = training batches, amber = the 2,000 held-out test images."):
                    loss_plot = ui.plotly(EMPTY).classes("w-full")
                with Panel("Accuracy", "Share of digits classified correctly. Training is measured on the batches it just learned from; test accuracy is on images it never trained on."):
                    acc_plot = ui.plotly(EMPTY).classes("w-full")

            # ------------------------------------------------------------ column 2: confusion + look inside
            with ui.column().classes("w-full gap-3"):
                with Panel("Confusion matrix", "Row = the true digit, column = what the network predicted. A perfect model lights up only the diagonal. Bright off-diagonal "
                                               "cells show which digits get mixed up (4↔9 and 3↔5 are classic)."):
                    conf_plot = ui.plotly(EMPTY).classes("w-full")
                with Panel("Most confident mistakes", "Test images the network got wrong while being most sure of itself. Click one to look inside the network at it."):
                    worst_box = ui.row().classes("gap-2 flex-wrap")
                with Panel("Layer-1 filters", "What the first layer looks for. In a CNN each 5×5 tile is a small pattern (edge, stroke, curve) slid over the image; red = bright pixels "
                                              "excite it, blue = they inhibit it. For the MLP these are the first 16 of its 128 neurons, each seeing the whole image."):
                    filters_plot = ui.plotly(EMPTY).classes("w-full")

            # ------------------------------------------------------------ column 3: inspect + draw
            with ui.column().classes("w-full gap-3"):
                with Panel("Try it: draw a digit", "Draw with the mouse or finger. Your drawing is shrunk to 20×20, centred by its centre of mass and sent through the network, "
                                                   "exactly like a real MNIST digit. Train the network first: an untrained one guesses at random."):
                    with ui.row().classes("gap-3 items-start no-wrap"):
                        pad = ui.interactive_image(data_uri(canvas), cross=False, events=["mousedown", "mousemove", "mouseup", "mouseleave"]).classes("border")
                        pad.style(f"width:{CANVAS}px;height:{CANVAS}px;max-width:100%;cursor:crosshair;touch-action:none;border-color:var(--ls-line)")
                        with ui.column().classes("gap-1"):
                            ui.label("the network sees").classes("text-[10px] ls-muted")
                            seen = ui.image(digit_uri(np.zeros((28, 28)), 4)).classes("w-[112px] h-[112px]").style("image-rendering: pixelated")
                            ui.button("Clear", icon="backspace", on_click=lambda: clear_pad()).props("outline dense no-caps")
                            ui.button("Random test digit", icon="casino", on_click=lambda: pick_test(int(np.random.randint(run.data.n_test)))).props("outline dense no-caps")
                with Panel("What it thinks", "The network's probability for each digit 0–9 for the image above (teal = its choice). The dotted amber box marks the true label when it is a test image."):
                    verdict = ui.label("–").classes("text-lg font-bold")
                    probs_plot = ui.plotly(EMPTY).classes("w-full")
                with Panel("Inside the network", "Feature maps: the response of each filter to the image above. Layer 1 finds small strokes; layer 2 combines them into bigger shapes. "
                                                 "Each tile is normalised separately (brighter = stronger response)."):
                    inside_label = ui.label().classes("text-xs ls-muted")
                    maps1 = ui.plotly(EMPTY).classes("w-full")
                    maps2 = ui.plotly(EMPTY).classes("w-full")

    # ------------------------------------------------------------------ interactions
    def clear_pad() -> None:
        canvas[:] = 0
        st.last, st.drawing, st.drawn, st.canvas_dirty = None, False, False, True
        pad.set_source(data_uri(canvas))
        seen.set_source(digit_uri(np.zeros((28, 28)), 4))
        mark_dirty()

    def classify_pad() -> None:
        img = to_mnist(canvas)
        seen.set_source(digit_uri(img, 4))
        run.focus_pixels(img)
        st.drawn = True

    def pick_test(i: int) -> None:
        st.drawn = False
        canvas[:] = 0
        pad.set_source(data_uri(canvas))
        seen.set_source(digit_uri(run.data.test_x[i, 0], 4))
        run.focus_test(i)
        mark_dirty()

    def on_mouse(e) -> None:
        pos = (float(e.image_x), float(e.image_y))
        if e.type == "mousedown":
            st.drawing, st.last = True, None
            stroke(canvas, None, pos)
            st.last, st.canvas_dirty = pos, True
        elif e.type == "mousemove" and st.drawing:
            if not getattr(e, "buttons", 1):  # released outside the pad
                st.drawing, st.last = False, None
                classify_pad()
                return
            stroke(canvas, st.last, pos)
            st.last, st.canvas_dirty = pos, True
        elif e.type in ("mouseup", "mouseleave") and st.drawing:
            st.drawing, st.last = False, None
            classify_pad()

    pad.on_mouse(on_mouse)

    # ------------------------------------------------------------------ refresh loop
    def refresh_pad() -> None:
        if st.canvas_dirty:
            st.canvas_dirty = False
            pad.set_source(data_uri(canvas))
            if st.drawing:
                classify_pad()

    def refresh() -> None:
        running = run.running
        play.props(f'icon={"pause" if running else "play_arrow"}')
        play.set_text("Pause" if running else "Train")
        error_label.set_text(run.error or "")
        if dark.value != st.dark:
            st.dark = dark.value
            mark_dirty()
        if run.version == st.version and not running:
            return
        with run.lock:
            st.version = run.version
            tr, pts, ev, focus = run.trainer, list(run.points), run.last_eval, run.focus
            label, image = run.focus_label, run.focus_image
            filters = tr.filters()
            step, epoch = tr.step, tr.step * tr.cfg.batch_size / tr.data.n_train
            n_params, model = tr.net.count_params(), run.cfg.model
        t_step.set_text(str(step))
        t_epoch.set_text(f"{epoch:.2f}")
        last = pts[-1] if pts else None
        t_tl.set_text(f"{last.train_loss:.3f}" if last else "–")
        t_ta.set_text(f"{last.train_acc * 100:.1f}%" if last else "–")
        t_va.set_text(f"{ev.test_acc * 100:.1f}%" if ev else "–")
        t_ips.set_text(f"{run.images_per_second:,.0f}" if step else "–")
        t_par.set_text(f"{n_params:,}")

        loss_plot.update_figure(charts.mnist_curves_figure(pts, "loss", st.dark))
        acc_plot.update_figure(charts.mnist_curves_figure(pts, "acc", st.dark))
        if ev is not None:
            conf_plot.update_figure(charts.confusion_figure(ev.confusion, st.dark))
            sig = tuple((w["index"], w["pred"]) for w in ev.worst)
            if sig != st.worst_sig:
                st.worst_sig = sig
                worst_box.clear()
                with worst_box:
                    if not ev.worst:
                        ui.label("No mistakes yet, or not trained yet.").classes("text-xs ls-muted")
                    for w in ev.worst[:8]:
                        with ui.column().classes("items-center gap-0 cursor-pointer").on("click", lambda w=w: pick_test(w["index"])):
                            ui.image(digit_uri(run.data.test_x[w["index"], 0], 3)).classes("w-[56px] h-[56px]").style("image-rendering: pixelated")
                            ui.label(f"{w['truth']}→{w['pred']}").classes("num text-[11px] text-red-400")
        filters_plot.update_figure(charts.mosaic_figure(filters, 4 if model == "cnn" else 8, True, st.dark, 190 if model == "cnn" else 200))

        if focus is not None:
            probs_plot.update_figure(probs_figure(focus.probs, label, st.dark))
            pred = int(np.argmax(focus.probs))
            source = "your drawing" if st.drawn else f"test image #{getattr(run, 'focus_index', 0)} (true label {label})"
            verdict.set_text(f"{pred}   ({focus.probs[pred] * 100:.0f}% sure) · {source}")
            if not st.drawn and image is not None:
                seen.set_source(digit_uri(image, 4))
            if focus.conv1 is not None:
                inside_label.set_text("conv layer 1 (8 maps, 24×24) and layer 2 (16 maps, 10×10)")
                maps1.update_figure(charts.mosaic_figure(focus.conv1, 4, False, st.dark, 150))
                maps2.update_figure(charts.mosaic_figure(focus.conv2, 8, False, st.dark, 110))
                maps2.set_visibility(True)
            else:
                inside_label.set_text("first hidden layer of the MLP: 128 neurons (16 × 8 grid)")
                maps1.update_figure(charts.mosaic_figure(focus.hidden.reshape(1, 8, 16), 1, False, st.dark, 110))
                maps2.set_visibility(False)

    ui.timer(0.5, refresh)
    ui.timer(0.08, refresh_pad)
    return run
