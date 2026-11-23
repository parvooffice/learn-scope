# Architecture

## The one idea

**The UI never trains, and the trainer never draws.** Training happens on a background thread that appends *snapshots*
to a bounded history. Every view in the app is a pure function of one snapshot, so the same code draws the live head
and any point in the past. That is what makes time travel, the Model Doctor and the Compare page cheap.

```
 Browser (NiceGUI + Plotly)            Python process
 ┌───────────────────────┐  websocket  ┌────────────────────────────────────────────┐
 │ charts, controls,     │ ◀─────────▶ │ ui/*.py pages (ui.timer refresh, 0.25 s)   │
 │ draw pad              │             │   reads ▼              commands ▲          │
 └───────────────────────┘             │ runner.Run  ── background thread ──┐       │
                                       │   HistoryBuffer of Snapshots       ▼       │
                                       │ Doctor(rules) ◀── Trainer (NumPy backprop) │
                                       └────────────────────────────────────────────┘
```

## Layers

| Package / module                      | Responsibility                                                                                                                         |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `autograd.py`, `tinynet.py`           | Scalar reverse-mode autograd (`Value`) and a tiny MLP built on it. Used by the Backprop page and as a teaching reference.              |
| `explainer.py`                        | The 2-3-1 network behind the Backprop page: autograd gradients **and** the chain rule written out by hand, compared on every call.     |
| `activations.py`, `mlp.py`            | Vectorised NumPy MLP: forward, hand-derived backward, SGD / momentum / Adam, BCE-with-logits (stable), L2.                             |
| `trainer.py`                          | One run's training state: data batches, the optimizer, live-adjustable lr / L2 / batch size, snapshot creation.                        |
| `snapshots.py`, `stats.py`            | The snapshot record (weights, per-layer gradient / activation / weight statistics, metrics) and `HistoryBuffer`.                       |
| `forward.py`, `landscape.py`          | Forward passes from stored weights (decision boundary, per-neuron maps) and the 2-D loss-landscape slice.                              |
| `doctor/`                             | Seven pure rules over a list of snapshots. Thresholds are module constants so tests and readers see exactly what triggers a rule.      |
| `runner.py`                           | `Run`: owns a `Trainer`, a history and a background thread; commands (play, pause, step, reset, configure, time travel).               |
| `mnist/`                              | `layers.py` (im2col Conv2D, MaxPool2, Dense, ReLU, Flatten, softmax-CE, Adam), `trainer.py` (`MnistTrainer`, `MnistRun`), `draw.py`.  |
| `config.py`                           | Frozen dataclasses for an experiment, sanitising of untrusted input and the URL encoding.                                              |
| `ui/`                                 | `charts.py` (pure data → Plotly dict), `components.py` (frame, panels, settings), and one module per page.                             |
| `app.py`                              | Routes and server start-up.                                                                                                            |

`ui/charts.py` has no NiceGUI imports, so every chart is testable without a browser.

## Decisions and why

**Snapshots store weights, and views are recomputed from them.** A snapshot is ~3 KB for the default network. The
decision boundary, per-neuron maps and loss-landscape path are recomputed by `forward.py` from those weights, so scrubbing
to step 40 shows exactly what the network looked like at step 40, and nothing else needs storing.

**`HistoryBuffer` thins instead of dropping.** At the item or byte limit it keeps every second snapshot of the older
half. The beginning of training (where the interesting failures are) is never thrown away; only its resolution drops.

**The Doctor is rule-based.** Deterministic rules are explainable (each returns evidence with a step to jump to),
unit-testable on synthetic histories and cheap. The cost: thresholds were tuned on a handful of runs.

**The Doctor is throttled, not debounced.** The first version debounced: "run 250 ms after the last snapshot". But
snapshots arrive every ~30 ms, so the quiet period never came and the Doctor never ran while training. Now it runs at
most every 0.25 s regardless of traffic (`DOCTOR_INTERVAL`). `tests/test_runner.py` has the regression test, and was
verified to fail on the old code.

**Figures are plain dicts, not `plotly.graph_objects`.** Building validated `Figure` objects cost 40-60 ms each; with
nine charts per refresh that starved the training thread (25 steps/s). Plain `{"data": [...], "layout": {...}}` dicts
cost microseconds, plots are updated in place, and the playground went to ~86 steps/s with the UI attached. A test guards
the refresh budget.

**The GIL is the real constraint.** Training (NumPy releases the GIL inside big operations, but tiny networks spend their
time in Python) and the UI share one interpreter. `sys.setswitchinterval(0.001)` makes the trainer thread get the GIL back
quickly while the UI thread builds charts.

**Config is data.** An experiment is a frozen dataclass, so it can be compared, copied, put in a URL and used as a
dictionary key. Anything from a URL goes through `sanitize_config`: unknown values fall back to defaults, numbers are
clamped, layer counts are capped and the string length is limited before decoding, so a hostile link cannot crash the
app or request absurd sizes.

**Learning rate, L2 and batch size are live; everything else restarts.** Changing the dataset or the architecture
changes the shape of the problem, so it rebuilds the run (`LIVE_KEYS` in `config.py`).

**MNIST is NumPy end to end.** Convolution uses im2col (one matrix multiply per layer) and the backward pass uses
col2im. Max-pool splits the gradient across ties. All layers are checked against finite differences. The data is a
10,000 / 2,000 image subset shipped as a 1.9 MB `.npz`, so no download happens at run time.

**The loss landscape** follows Li et al.: two random directions, each rescaled per layer to that layer's weight norm, made
orthogonal so that projecting any weight vector onto the plane gives exact coordinates (tested).

## Threading rules

- One `Run` per open page; it is closed when the browser disconnects (`on_disconnect`), which stops its thread.
- All shared state sits behind one `RLock` inside `Run` / `MnistRun`. UI code reads through methods that copy under the lock
  (`series()`, `view_snapshot()`), never the live arrays.
- The UI polls `run.version` and only redraws when it changed or the run is playing.
- The loss-landscape slice (about a second) runs in `nicegui.run.io_bound` so the page stays responsive.

## Known limits

- CPU only; fine for tiny networks and a 10k-image MNIST subset, not for large models.
- One training thread per open tab: serving many viewers needs a bigger machine.
- Redraws a few times per second over a websocket, so it is built for understanding rather than for 60 fps animation.
- Tiny networks are init-sensitive (see the seed note in the README).
