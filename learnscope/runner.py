"""A training run that lives on a background thread.

The UI never trains: it sends commands to a `Run` (play, pause, step, reset, change settings) and
reads the recorded history. Every snapshot is kept in a bounded `HistoryBuffer`, which is what makes
time travel possible: any view can be redrawn from any recorded step.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_CONFIG, LIVE_KEYS, PlaygroundConfig
from .datasets import Dataset, generate_dataset, split_dataset
from .doctor import Diagnosis, DoctorContext, Fix, diagnose
from .snapshots import HistoryBuffer, Snapshot, snapshot_bytes
from .trainer import Trainer

TICK_SECONDS = 0.033  # at most ~30 snapshots per second
DOCTOR_INTERVAL = 0.25  # the Doctor runs at most 4 times a second (a throttle, not a debounce)


@dataclass
class Series:
    """Arrays for the charts, copied under the lock so the UI can use them freely."""

    steps: np.ndarray
    train_loss: np.ndarray
    test_loss: np.ndarray
    train_acc: np.ndarray
    test_acc: np.ndarray
    g_norms: np.ndarray  # (snapshots, layers)


class Run:
    def __init__(self, config: PlaygroundConfig = DEFAULT_CONFIG, speed: int = 4) -> None:
        self.lock = threading.RLock()
        self.speed = speed
        self.history: HistoryBuffer[Snapshot] = HistoryBuffer(snapshot_bytes, 3000, 150 * 1024 * 1024)
        self.version = 0  # bumps on every change the UI should redraw
        self.selected_step: int | None = None  # None = follow the live head
        self.diagnoses: list[Diagnosis] = []
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_doctor = 0.0
        self._build(config)

    # ------------------------------------------------------------------ construction
    def _build(self, cfg: PlaygroundConfig) -> None:
        self.pause()
        with self.lock:
            self.config = cfg
            ds = generate_dataset(cfg.dataset, cfg.samples, cfg.noise, cfg.seed)
            self.train, self.test = split_dataset(ds, cfg.test_split, cfg.seed)
            self.trainer = Trainer(cfg, self.train, self.test)
            self.history.clear()
            self.history.push(self.trainer.snapshot())
            self.selected_step = None
            self.diagnoses = []
            self.error = None
            self.version += 1

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def specs(self):
        return self.trainer.specs

    # ------------------------------------------------------------------ commands
    def start(self) -> None:
        with self.lock:
            self.selected_step = None
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="learnscope-train", daemon=True)
            self._thread.start()

    def pause(self) -> None:
        t = self._thread
        if t is not None and t.is_alive():
            self._stop.set()
            if t is not threading.current_thread():
                t.join(timeout=5)
        self._thread = None
        self._stop.clear()

    def toggle(self) -> None:
        self.pause() if self.running else self.start()

    def step(self, n: int = 1) -> None:
        self.pause()
        with self.lock:
            self.selected_step = None
            for _ in range(max(1, n)):
                self.trainer.train_step()
            self._record()

    def reset(self) -> None:
        self._build(self.config)

    def configure(self, cfg: PlaygroundConfig, keep_running: bool = True) -> None:
        """Rebuild with a new configuration (data, architecture, optimizer...). Restarts training."""
        was_running = self.running
        self._build(cfg)
        if was_running and keep_running:
            self.start()

    def set_config(self, **patch) -> None:
        """Change settings. lr / l2 / batch size apply instantly; anything else restarts the run."""
        if all(k in LIVE_KEYS for k in patch):
            with self.lock:
                self.config = self.config.with_(**patch)
                self.trainer.set_live(patch.get("lr"), patch.get("l2"), patch.get("batch_size"))
                self.version += 1
        else:
            self.configure(self.config.with_(**patch))

    def set_speed(self, steps_per_tick: int) -> None:
        self.speed = max(1, int(steps_per_tick))

    def apply_fix(self, fix: Fix) -> None:
        cfg = fix.apply(self.config)
        if not fix.reset and all(k in LIVE_KEYS for k in fix.patch):
            self.set_config(**fix.patch)
            self._run_doctor(force=True)
        else:
            self._build(cfg)
            self.start()

    # ------------------------------------------------------------------ time travel
    def scrub_to(self, index: int) -> None:
        self.pause()
        with self.lock:
            snap = self.history.at(index)
            if snap is not None:
                self.selected_step = snap.step
                self.version += 1

    def jump_to_step(self, step: int) -> None:
        with self.lock:
            i = self.history.nearest(step, lambda s: s.step)
        if i >= 0:
            self.scrub_to(i)

    def go_live(self) -> None:
        with self.lock:
            self.selected_step = None
            self.version += 1

    def nudge(self, delta: int) -> None:
        with self.lock:
            n = len(self.history)
            cur = n - 1 if self.selected_step is None else self.history.nearest(self.selected_step, lambda s: s.step)
            nxt = max(0, min(n - 1, cur + delta))
            if delta > 0 and nxt == n - 1 and self.selected_step is None:
                return
        self.scrub_to(nxt)

    @property
    def selected_index(self) -> int:
        with self.lock:
            if self.selected_step is None:
                return len(self.history) - 1
            return self.history.nearest(self.selected_step, lambda s: s.step)

    def view_snapshot(self) -> Snapshot | None:
        """The snapshot every view should draw: the live head, or the time-travel target."""
        with self.lock:
            if self.selected_step is None:
                return self.history.last()
            return self.history.at(self.history.nearest(self.selected_step, lambda s: s.step))

    def series(self) -> Series:
        with self.lock:
            snaps = self.history.to_list()
        n_layers = len(snaps[-1].layers) if snaps else 0
        return Series(
            steps=np.array([s.step for s in snaps]),
            train_loss=np.array([s.train_loss for s in snaps]),
            test_loss=np.array([s.test_loss for s in snaps]),
            train_acc=np.array([s.train_acc for s in snaps]),
            test_acc=np.array([s.test_acc for s in snaps]),
            g_norms=np.array([[l.g_norm for l in s.layers] for s in snaps]).reshape(len(snaps), n_layers),
        )

    # ------------------------------------------------------------------ internals
    def _record(self) -> None:
        self.history.push(self.trainer.snapshot())
        self.version += 1
        self._run_doctor()

    def _run_doctor(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_doctor < DOCTOR_INTERVAL:
            return
        self._last_doctor = now
        self.diagnoses = diagnose(DoctorContext(self.history.to_list(), self.config))

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                with self.lock:
                    for _ in range(self.speed):
                        self.trainer.train_step()
                    self._record()
            except Exception as exc:  # noqa: BLE001 - surfaced in the UI, never a silent thread death
                self.error = f"{type(exc).__name__}: {exc}"
                self._stop.set()
                break
            self._stop.wait(max(0.0, TICK_SECONDS - (time.perf_counter() - t0)))

    def close(self) -> None:
        self.pause()


def datasets_for(cfg: PlaygroundConfig) -> tuple[Dataset, Dataset]:
    ds = generate_dataset(cfg.dataset, cfg.samples, cfg.noise, cfg.seed)
    return split_dataset(ds, cfg.test_split, cfg.seed)
