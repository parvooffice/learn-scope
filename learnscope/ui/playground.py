"""The Playground page: train, watch, rewind, diagnose."""

from __future__ import annotations

from types import SimpleNamespace

from nicegui import run as nrun
from nicegui import ui

from ..config import DEFAULT_CONFIG, PlaygroundConfig, encode_config
from ..forward import forward_grid
from ..landscape import compute_landscape, flatten, project
from ..runner import Run
from . import charts
from .components import (
    Panel,
    diagnosis_signature,
    doctor_cards,
    fmt_lr,
    frame,
    settings_panel,
    stat_tile,
)

EMPTY = {"data": [], "layout": {}}
SPEEDS = {1: "1×", 4: "4×", 16: "16×", 64: "64×"}


def build(cfg: PlaygroundConfig = DEFAULT_CONFIG, autoplay: bool = True) -> Run:
    dark = frame("Playground")
    run = Run(cfg, speed=4)
    ui.context.client.on_disconnect(run.close)
    st = SimpleNamespace(version=-1, dark=dark.value, neuron=None, landscape=None, diag_sig=None, selected_index=-1, flash=None)

    def mark_dirty() -> None:
        st.version = -1

    with ui.element("div").classes("w-full flex flex-col xl:flex-row items-stretch xl:items-start gap-3 p-3"):
        # ---------------------------------------------------------------- left: settings
        with ui.column().classes("w-full xl:w-72 gap-3 xl:shrink-0 order-2 xl:order-1"):
            settings_panel(run, on_change=lambda: (setattr(st, "landscape", None), setattr(st, "neuron", None), mark_dirty()))

        # ---------------------------------------------------------------- centre
        with ui.column().classes("w-full xl:flex-1 min-w-0 gap-3 order-1 xl:order-2"):
            with ui.row().classes("items-center gap-2 w-full"):
                play = ui.button("Play", icon="play_arrow", on_click=lambda: (run.toggle(), mark_dirty())).props("color=primary no-caps")
                ui.button("Step", on_click=lambda: (run.step(1), mark_dirty())).props("outline no-caps")
                ui.button("Step ×10", on_click=lambda: (run.step(10), mark_dirty())).props("outline no-caps")
                ui.button("Reset", icon="restart_alt", on_click=lambda: (run.reset(), mark_dirty())).props("outline no-caps")
                ui.label("speed").classes("text-xs ls-muted ml-2")
                ui.toggle(SPEEDS, value=4, on_change=lambda e: run.set_speed(e.value)).props("dense no-caps size=sm")
                ui.space()
                ui.badge("NumPy · CPU", color="grey").props("outline").tooltip("Pure-Python engine: tiny networks train fast on a CPU")

                def share() -> None:
                    base = str(ui.context.client.request.base_url)
                    ui.clipboard.write(f"{base}playground?c={encode_config(run.config)}")
                    ui.notify("Link copied: opens this exact experiment", type="positive")

                ui.button("Share", icon="link", on_click=share).props("outline no-caps")
            error_label = ui.label().classes("text-red-400 text-sm")

            with ui.row().classes("gap-2"):
                t_step = stat_tile("step")
                t_tl = stat_tile("train loss", "text-teal-400")
                t_vl = stat_tile("test loss", "text-amber-400")
                t_ta = stat_tile("train acc")
                t_va = stat_tile("test acc")
                t_lr = stat_tile("learning rate")

            with ui.element("div").classes("w-full flex flex-col lg:flex-row gap-3 items-stretch lg:items-start"):
                with Panel("Decision boundary", "The heat map is the network's prediction at every point of the plane: purple = confident class 0, yellow = confident class 1; "
                                                "the boundary is where the colour switches. Dots are training points; rings are test points the network never trained on.",
                           classes="w-full lg:w-[380px] lg:shrink-0") as boundary_panel:
                    boundary = ui.plotly(charts.boundary_figure(None, run.train, run.test, st.dark)).classes("w-full")
                with ui.column().classes("w-full lg:flex-1 min-w-0 gap-3"):
                    with Panel("Network", "Circles are neurons and lines are weights (red positive, blue negative, thicker = larger). A neuron's colour is how active it is on "
                                          "average; a neuron drawn as × is a dead ReLU. Click a hidden neuron to see what it responds to."):
                        network = ui.plotly(charts.network_figure(run.specs, None, None, st.dark)).classes("w-full")
                    neuron_holder = ui.column().classes("w-full")
                    with neuron_holder:
                        with Panel("Neuron", "This neuron's own decision boundary: how strongly it fires at each point of the input plane. Each neuron carves out one simple shape; the layers after it combine those shapes into the final boundary."):
                            neuron_label = ui.label().classes("text-xs ls-muted")
                            neuron_plot = ui.plotly(EMPTY).classes("w-full")
                    neuron_holder.set_visibility(False)

            with Panel("Loss landscape", "The loss depends on every weight: far too many dimensions to draw. This picks two random directions in weight space and shows "
                                         "the loss on the flat slice through the weights at the step you are viewing (teal dot). Dark = low loss, bright = high. "
                                         "The white line is where training came from, projected onto the slice; it can leave the picture because the real path "
                                         "uses many more directions."):
                with ui.row().classes("items-center gap-2"):
                    compute_btn = ui.button("Compute", icon="terrain").props("outline dense no-caps")
                    land_note = ui.label("Press Compute (about a second) to slice the loss surface at the step you are viewing.").classes("text-xs ls-muted")
                land_plot = ui.plotly(EMPTY).classes("w-full")
                land_plot.set_visibility(False)

        # ---------------------------------------------------------------- right
        with ui.column().classes("w-full xl:w-[390px] gap-3 xl:shrink-0 order-3"):
            with Panel("Model Doctor", "The Doctor reads the telemetry of the run (gradient sizes, dead neurons, loss curves) and applies deterministic rules to spot common "
                                       "problems. “Show me” jumps back in time to where the problem is visible; the fix button applies a suggested change."):
                doctor_box = ui.column().classes("w-full gap-2")
            panels: dict[str, Panel] = {}
            with Panel("Loss", "Loss is one number for “how wrong the network is” (lower is better): teal on the training points, amber on held-out test points. "
                               "If the lines drift apart the network is memorising. Click the chart to time-travel to that step.") as p_loss:
                loss_plot = ui.plotly(charts.loss_figure(run.series(), None, st.dark)).classes("w-full")
            panels["loss"] = p_loss
            with Panel("Accuracy", "The fraction of points classified correctly, training (teal) and test (amber). 50% is a coin flip on these balanced datasets."):
                acc_plot = ui.plotly(charts.accuracy_figure(run.series(), None, st.dark)).classes("w-full")
            with Panel("Gradient size per layer", "The gradient tells each weight which way to move. This shows how large it is in each layer (log scale). If the early layers "
                                                  "(L1…) sit far below the output layer, gradients are vanishing and those layers barely learn.") as p_grad:
                grad_plot = ui.plotly(charts.gradient_figure(run.series(), None, st.dark)).classes("w-full")
            panels["grad"] = p_grad
            with Panel("Weight distributions", "A histogram of every weight in each layer at the selected step. Training starts with small random values and spreads them "
                                               "out; a very wide histogram means weights are growing large.") as p_w:
                w_plot = ui.plotly(EMPTY).classes("w-full")
            panels["weights"] = p_w
            with Panel("Activations per layer", "Each neuron's output on the training points. For ReLU layers a tall bar at 0 and a high “dead” percentage means many neurons "
                                                "stopped firing for good. For tanh/sigmoid, bars piled at the ends mean saturated neurons.") as p_a:
                a_plot = ui.plotly(EMPTY).classes("w-full")
            panels["activations"] = p_a
            panels["network"] = boundary_panel

    # -------------------------------------------------------------------- timeline
    with ui.row().classes("w-full items-center gap-3 px-4 py-2 border-t").style("border-color: var(--ls-line)"):
        for icon, delta, tip in (("keyboard_double_arrow_left", -10, "Back 10 snapshots"), ("chevron_left", -1, "Previous snapshot (←)")):
            ui.button(icon=icon, on_click=lambda d=delta: (run.nudge(d), mark_dirty())).props("flat dense round").tooltip(tip)
        ui.button(icon="chevron_right", on_click=lambda: (run.nudge(1), mark_dirty())).props("flat dense round").tooltip("Next snapshot (→)")
        ui.button(icon="keyboard_double_arrow_right", on_click=lambda: (run.nudge(10), mark_dirty())).props("flat dense round")
        timeline = ui.slider(min=0, max=1, step=1, value=0).classes("flex-1").props('dense aria-label="Timeline scrubber: drag to view the network at any recorded training step"')
        timeline.on("update:model-value", lambda e: (run.scrub_to(int(e.args)), mark_dirty()))
        time_label = ui.label("step 0").classes("num text-xs")
        live_badge = ui.badge("LIVE", color="primary").props("outline")
        live_btn = ui.button("Back to live", on_click=lambda: (run.go_live(), mark_dirty())).props("dense no-caps color=primary")
        from .components import info_button

        info_button("Every few training steps the app records a snapshot: all weights, gradient sizes, activation statistics and metrics. Drag the slider "
                    "(or press ← →) to rewind: the network graph, decision boundary and histograms all redraw exactly as they were at that moment.",
                    "What is the timeline?")

    # -------------------------------------------------------------------- interactions
    async def on_neuron_click(e) -> None:
        try:
            layer, unit = e.args["points"][0]["customdata"]
        except (KeyError, IndexError, TypeError, ValueError):
            return
        st.neuron = (int(layer), int(unit)) if layer >= 0 and layer < len(run.specs) - 1 else None
        mark_dirty()

    network.on("plotly_click", on_neuron_click)
    loss_plot.on("plotly_click", lambda e: (run.jump_to_step(int(e.args["points"][0]["x"])), mark_dirty()) if e.args.get("points") else None)

    async def compute_landscape_now() -> None:
        snap = run.view_snapshot()
        if snap is None:
            return
        compute_btn.props("loading")
        land = await nrun.io_bound(compute_landscape, run.specs, snap.weights, run.train.x, run.train.y)
        compute_btn.props(remove="loading")
        st.landscape = (land, snap.step)
        mark_dirty()

    compute_btn.on_click(compute_landscape_now)

    def show_flash(d) -> None:
        p = panels.get(d.evidence.chart)
        if p:
            p.flash()

    @ui.refreshable
    def doctor_view(diagnoses) -> None:
        doctor_cards(run, diagnoses, show_flash)

    def on_key(e) -> None:
        if not e.action.keydown:
            return
        if e.key == " ":
            run.toggle()
        elif e.key.arrow_right:
            (run.step(1) if run.selected_step is None else run.nudge(1)) if not e.modifiers.shift else run.step(10)
        elif e.key.arrow_left:
            run.nudge(-1)
        elif e.key == "r":
            run.reset()
        elif e.key == "l":
            run.go_live()
        mark_dirty()

    ui.keyboard(on_key=on_key)

    # -------------------------------------------------------------------- refresh loop
    def refresh() -> None:
        running = run.running
        play.props(f'icon={"pause" if running else "play_arrow"}')
        play.set_text("Pause" if running else "Play")
        error_label.set_text(run.error or "")
        if dark.value != st.dark:
            st.dark = dark.value
            mark_dirty()
        if run.version == st.version and not running:
            return
        st.version = run.version
        snap = run.view_snapshot()
        if snap is None:
            return
        live = run.selected_step is None
        s = run.series()
        marker = None if live else snap.step

        t_step.set_text(str(snap.step))
        t_tl.set_text(f"{snap.train_loss:.4f}")
        t_vl.set_text(f"{snap.test_loss:.4f}")
        t_ta.set_text(f"{snap.train_acc * 100:.1f}%")
        t_va.set_text(f"{snap.test_acc * 100:.1f}%")
        t_lr.set_text(fmt_lr(snap.lr))

        grid = forward_grid(run.specs, snap.weights)
        boundary.update_figure(charts.boundary_figure(grid[-1][:, :, 0], run.train, run.test, st.dark))
        network.update_figure(charts.network_figure(run.specs, snap, st.neuron, st.dark))
        loss_plot.update_figure(charts.loss_figure(s, marker, st.dark))
        acc_plot.update_figure(charts.accuracy_figure(s, marker, st.dark))
        grad_plot.update_figure(charts.gradient_figure(s, marker, st.dark))

        show_neuron = st.neuron is not None and st.neuron[0] < len(run.specs) - 1
        neuron_holder.set_visibility(show_neuron)
        if show_neuron:
            neuron_label.set_text(f"layer {st.neuron[0] + 1} · unit {st.neuron[1] + 1}")
            neuron_plot.update_figure(charts.neuron_figure(grid[st.neuron[0]], st.neuron[1], st.dark))

        n = len(snap.layers)
        names = [("output" if i == n - 1 else f"layer {i + 1}") for i in range(n)]
        w_plot.update_figure(charts.layer_hist_figure(
            [l.w_hist for l in snap.layers],
            [f"{names[i]}   μ {l.w_mean:.2f} · σ {l.w_std:.2f}" for i, l in enumerate(snap.layers)], charts.LAYER_COLORS, st.dark))
        a_plot.update_figure(charts.layer_hist_figure(
            [l.a_hist for l in snap.layers],
            [f"{names[i]} · {run.specs[i].activation}   "
             + (f"dead {l.dead_pct * 100:.0f}%" if run.specs[i].activation in ("relu", "gelu") else f"saturated {l.sat_pct * 100:.0f}%")
             for i, l in enumerate(snap.layers)], charts.LAYER_COLORS, st.dark, zero_line=False))

        # timeline
        count = len(run.history)
        timeline.props(f"max={max(1, count - 1)}")
        timeline.value = run.selected_index
        time_label.set_text(f"step {snap.step} · epoch {snap.epoch:.1f} · {count} snapshots · {run.history.total_bytes / 1048576:.1f} MB")
        live_badge.set_visibility(live)
        live_btn.set_visibility(not live)

        # doctor
        sig = diagnosis_signature(run.diagnoses)
        if sig != st.diag_sig:
            st.diag_sig = sig
            doctor_box.clear()
            with doctor_box:
                doctor_view(run.diagnoses)

        # loss landscape (computed on demand; the path follows the time-travel position)
        if st.landscape is not None:
            land, at_step = st.landscape
            snaps = run.history.to_list()
            center = flatten(next((x for x in snaps if x.step == at_step), snap).weights)
            stride = max(1, len(snaps) // 300)
            path = [project(flatten(x.weights), center, land.d1, land.d2) for x in snaps[::stride] if x.step <= at_step]
            land_plot.update_figure(charts.landscape_figure(land, path, st.dark))
            land_plot.set_visibility(True)
            land_note.set_text(f"slice at step {at_step} · loss at centre {land.center_loss:.3f} · colour range {land.min:.2f}–{land.clip:.2f}")
        else:
            land_plot.set_visibility(False)
            land_note.set_text("Press Compute (about a second) to slice the loss surface at the step you are viewing.")

    ui.timer(0.25, refresh)
    if autoplay:
        run.start()
    return run
