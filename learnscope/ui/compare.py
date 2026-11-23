"""Compare page: two runs on identical data, side by side. Change one setting and see exactly what it does."""

from __future__ import annotations

from types import SimpleNamespace

from nicegui import ui

from ..config import DEFAULT_CONFIG, PlaygroundConfig, encode_config
from ..forward import forward_grid
from ..runner import Run
from . import charts
from .components import Panel, frame, settings_panel, stat_tile

DEFAULT_B = DEFAULT_CONFIG.with_(optimizer="sgd", lr=0.1)
SPEEDS = {1: "1×", 4: "4×", 16: "16×"}


def verdict(a: Run, b: Run) -> str:
    sa, sb = a.view_snapshot(), b.view_snapshot()
    if sa is None or sb is None or min(sa.step, sb.step) < 30:
        return "Press Play on both runs. Every setting except the one you change stays identical, so any difference is caused by that change."
    da, db = sa.test_loss - sb.test_loss, sb.test_acc - sa.test_acc
    if abs(da) < 0.01 and abs(db) < 0.01:
        return f"Too close to call at step {min(sa.step, sb.step)}: test loss {sa.test_loss:.3f} vs {sb.test_loss:.3f}."
    lead = "A" if da < 0 else "B"
    w, l = (sa, sb) if lead == "A" else (sb, sa)
    return (f"Run {lead} is ahead at step {min(sa.step, sb.step)}: test loss {w.test_loss:.3f} vs {l.test_loss:.3f}, "
            f"test accuracy {w.test_acc * 100:.1f}% vs {l.test_acc * 100:.1f}%.")


def build(cfg_a: PlaygroundConfig | None = None, cfg_b: PlaygroundConfig | None = None) -> tuple[Run, Run]:
    dark = frame("Compare")
    a, b = Run(cfg_a or DEFAULT_CONFIG, speed=4), Run(cfg_b or DEFAULT_B, speed=4)
    ui.context.client.on_disconnect(lambda: (a.close(), b.close()))
    st = SimpleNamespace(dark=dark.value, ver=(-1, -1))

    def shared_data(**patch) -> None:
        was = a.running or b.running
        for r in (a, b):
            r.configure(r.config.with_(**patch), keep_running=False)
        if was:
            a.start()
            b.start()

    def mark_dirty() -> None:
        st.ver = (-1, -1)

    with ui.column().classes("w-full p-3 gap-3 max-w-[1500px] mx-auto"):
        ui.label("Compare two runs").classes("text-xl font-bold")
        ui.label("Both networks train on the same data and start from the same random seed. Defaults: Adam (A) vs plain SGD (B). Change anything on either side.").classes("text-sm ls-muted")
        with ui.row().classes("items-center gap-2 w-full"):
            play = ui.button("Play both", icon="play_arrow", on_click=lambda: toggle()).props("color=primary no-caps")
            ui.button("Step both", on_click=lambda: (a.step(1), b.step(1), mark_dirty())).props("outline no-caps")
            ui.button("Reset both", icon="restart_alt", on_click=lambda: (a.reset(), b.reset(), mark_dirty())).props("outline no-caps")
            ui.label("speed").classes("text-xs ls-muted ml-2")
            ui.toggle(SPEEDS, value=4, on_change=lambda e: (a.set_speed(e.value), b.set_speed(e.value))).props("dense no-caps size=sm")
            ui.space()

            def share() -> None:
                base = str(ui.context.client.request.base_url)
                ui.clipboard.write(f"{base}compare?a={encode_config(a.config)}&b={encode_config(b.config)}")
                ui.notify("Link copied: opens this exact comparison", type="positive")

            ui.button("Share", icon="link", on_click=share).props("outline no-caps")
        verdict_label = ui.label().classes("text-sm font-bold text-teal-400")
        error_label = ui.label().classes("text-red-400 text-sm")

        with ui.element("div").classes("w-full grid grid-cols-1 xl:grid-cols-[260px_1fr_260px] gap-3 items-start"):
            with ui.column().classes("w-full gap-3"):
                ui.label("RUN A (teal)").classes("text-xs font-bold tracking-wider text-teal-400")
                settings_panel(a, data=True, on_change=mark_dirty, shared_data_setter=shared_data)

            with ui.column().classes("w-full gap-3 min-w-0"):
                with ui.element("div").classes("w-full grid grid-cols-1 md:grid-cols-2 gap-3"):
                    with Panel("Run A: decision boundary", "Where run A's network currently puts the line between the two classes.", classes="min-w-0 overflow-hidden"):
                        with ui.row().classes("gap-2"):
                            a_step, a_loss, a_acc = stat_tile("step"), stat_tile("test loss", "text-amber-400"), stat_tile("test acc")
                        a_plot = ui.plotly(charts.boundary_figure(None, a.train, a.test, st.dark, 300)).classes("w-full")
                    with Panel("Run B: decision boundary", "Where run B's network currently puts the line between the two classes.", classes="min-w-0 overflow-hidden"):
                        with ui.row().classes("gap-2"):
                            b_step, b_loss, b_acc = stat_tile("step"), stat_tile("test loss", "text-amber-400"), stat_tile("test acc")
                        b_plot = ui.plotly(charts.boundary_figure(None, b.train, b.test, st.dark, 300)).classes("w-full")
                with Panel("Loss", "Run A in teal, run B in violet; solid = training data, dashed = held-out test data. The faster and lower a curve falls, the better that setup learns."):
                    loss_plot = ui.plotly(charts.compare_figure(a.series(), b.series(), "loss", st.dark)).classes("w-full")
                with Panel("Accuracy", "The share of points classified correctly over time, same colours as above."):
                    acc_plot = ui.plotly(charts.compare_figure(a.series(), b.series(), "acc", st.dark)).classes("w-full")

            with ui.column().classes("w-full gap-3"):
                ui.label("RUN B (violet)").classes("text-xs font-bold tracking-wider text-violet-400")
                settings_panel(b, data=False, on_change=mark_dirty)

    def toggle() -> None:
        if a.running or b.running:
            a.pause()
            b.pause()
        else:
            a.start()
            b.start()
        mark_dirty()

    def refresh() -> None:
        running = a.running or b.running
        play.set_text("Pause both" if running else "Play both")
        play.props(f'icon={"pause" if running else "play_arrow"}')
        error_label.set_text(a.error or b.error or "")
        if dark.value != st.dark:
            st.dark = dark.value
            mark_dirty()
        ver = (a.version, b.version)
        if ver == st.ver and not running:
            return
        st.ver = ver
        sa, sb = a.view_snapshot(), b.view_snapshot()
        if sa is None or sb is None:
            return
        for snap, run, plot, tiles in ((sa, a, a_plot, (a_step, a_loss, a_acc)), (sb, b, b_plot, (b_step, b_loss, b_acc))):
            tiles[0].set_text(str(snap.step))
            tiles[1].set_text(f"{snap.test_loss:.3f}")
            tiles[2].set_text(f"{snap.test_acc * 100:.1f}%")
            plot.update_figure(charts.boundary_figure(forward_grid(run.specs, snap.weights)[-1][:, :, 0], run.train, run.test, st.dark, 300))
        series_a, series_b = a.series(), b.series()
        loss_plot.update_figure(charts.compare_figure(series_a, series_b, "loss", st.dark))
        acc_plot.update_figure(charts.compare_figure(series_a, series_b, "acc", st.dark))
        verdict_label.set_text(verdict(a, b))

    ui.timer(0.3, refresh)
    return a, b
