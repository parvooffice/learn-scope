# Architecture

## Overview

```
src/
  app/            pages: Landing (+Hero), playground/, backprop/, mnist/, About
  components/     Panel, Info ("What am I looking at?"), Slider, AutoCanvas, Header…
  viz/            canvas visualisations: LineChart, Histogram, DecisionBoundary, NetworkGraph,
                  NeuronPreview, MapGrid, ConfusionMatrix, DrawPad, DigitCanvas
  engine/
    tiny/         hand-written autograd (Value, MLP) + the backprop explainer's model
    cpu/          dependency-free MLP with manual backprop (landing hero, gradient oracle)
    tf/           TensorFlow.js: backend selection, playground Trainer, MnistTrainer, layer stats
    datasets/     toy generators, MNIST loader (+ IndexedDB cache)
    forward.ts    CPU forward pass used by the UI to draw boundaries/neuron previews from weights
  worker/         training.worker.ts, mnist.worker.ts and their typed message protocols
  doctor/         one file per rule in rules/, doctor.ts runs them, tests use synthetic histories
  store/          zustand: ui (theme, highlight), run (playground + history), mnist, view
  lib/            ring buffer, URL config encoding, colours, router, RNG, shortcuts, stats
```

## Data flow (playground)

```
 UI action ──send()──▶ worker (single message queue)
                         │  Trainer.trainStep() × stepsPerTick   (TF.js, GPU)
                         │  Trainer.snapshot(): metrics, full-batch gradient stats,
                         │                      activation stats, weights
 history ◀─rAF flush──── postMessage(snapshot, [weight buffers])   (transferable)
    │
    ├─ views read viewSnapshot(selectedStep): the live head, or the time-travel target
    └─ Doctor: diagnose(history) debounced 250 ms ─▶ diagnoses ─▶ evidence links
```

- **The worker owns the model.** The UI thread never touches TensorFlow.js, so training cannot block rendering. The TF.js bundle (~2 MB) is a lazy chunk fetched only when a page that needs it opens; the first-load JS is 236 KB.
- **A `built` message separates runs.** Everything before it belongs to the previous model, so late snapshots from an old run can never pollute a new run's history.
- **Snapshots arrive batched per animation frame** (`requestAnimationFrame` flush) so React re-renders at most once per frame, however fast the worker is.
- **Canvas, not DOM.** Charts and graphs draw to `<canvas>` (`AutoCanvas` handles resize and device-pixel-ratio). React only re-renders the small wrapper component when a new snapshot arrives.

## Time travel

`HistoryBuffer` (src/lib/ringBuffer.ts) holds snapshots under **two limits: item count and bytes (150 MB)**. When a limit is hit it
_thins the oldest half_ (keeps every second item) instead of dropping the oldest item, so you can still rewind to the very start of training
while recent history stays dense. `selectedStep === null` means "follow live".

## Decisions and deviations from the original brief (and why)

| Decision                                                                                                                                                      | Reason                                                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Decision boundary and neuron previews are computed on the main thread from the snapshot's weights** (`forward.ts`), not shipped as a grid in each snapshot. | The nets are tiny (≤16 units/layer), so a 48×48 forward pass is ~1–2 M multiply-adds. This keeps snapshots ~3 KB, makes time-travel exact for any step, and `forward.ts` is cross-checked against TF.js in a unit test.           |
| **Custom training loop + hand-written SGD/momentum/Adam** instead of `model.fit` for the playground.                                                          | We need the gradients, a live-adjustable learning rate, and per-layer statistics at every snapshot. `model.fit` hides all three.                                                                                                  |
| **Gradient statistics use the full training set** (one extra forward/backward per snapshot), not the last mini-batch.                                         | Mini-batch gradients are noisy; the Doctor's vanishing/exploding rules need a stable signal.                                                                                                                                      |
| **`tf.layers` for MNIST**, custom loop only for batching.                                                                                                     | Convolutions are the point here; `tf.layers` gives conv/pool/dense with less code, and `trainOnBatch` is enough.                                                                                                                  |
| **MNIST subset shipped with the app** (10k train + 2k test, 2.3 MB PNG, in `public/data`) and cached decoded in IndexedDB.                                    | Removes a runtime dependency on a third-party host and keeps the first load small. Regenerate with `scripts/make-mnist-subset.mjs` from the public "model-builder" sprite (`storage.googleapis.com/learnjs-data/model-builder/`). |
| **Hash routing** (`#/playground?c=…`).                                                                                                                        | Works on any static host (also when opened from `dist/` via a plain file server) with no rewrite rules; the config in the hash is never sent to a server.                                                                         |
| **No D3 DOM rendering**, only `d3-scale` / `d3-array`.                                                                                                        | Canvas + d3 scales keeps bundle small and redraws cheap.                                                                                                                                                                          |
| **Project folder must not contain `#`.**                                                                                                                      | Vite's dev server fails to serve files from such paths (URL fragment). The project was moved to a clean path.                                                                                                                     |
| **The saturation rule only fires while loss is still high.**                                                                                                  | During manual testing a converged spiral network legitimately saturates tanh units, and the rule produced a false alarm. Encoded in a unit test.                                                                                  |

## Model Doctor

Rules are pure functions `(history, config) → Diagnosis | null` in `src/doctor/rules/*.ts` with thresholds exported as constants.
`doctor.ts` runs them all and orders by severity. If the run has exploded, the downstream symptoms are suppressed so the root cause is shown alone.
A diagnosis has: id, severity, title, plain-English explanation, evidence `{step, chart, detail}` and an optional fix `{label, patch, reset}`.
A fix that only changes `lr`/`l2`/`batchSize` and doesn't need a restart is applied **live** to the running optimizer.

## Performance notes (measure, don't guess)

- First-load JS: 236 KB raw, 75 KB gzip (landing, router, hero). TF.js (~2 MB raw) loads lazily inside the workers.
- MNIST CNN: ~2,100 img/s ≈ 4.7 s per 10k-image epoch on the developer's WebGPU laptop (shown live on the MNIST page).
- The training loop yields at least every 33 ms (playground) so the main thread keeps rendering; the UI does one React update per animation frame.
- Snapshot size is ~3 KB at the default architecture (observed: 422 snapshots ≈ 1.2 MB), so the 150 MB limit is far away for toy networks.
- UI frame rate while training the default spiral config: **60 fps, p95 frame 16.9 ms, no long tasks** (headless Chrome, WebGPU, 1440×900; reproduce with `npm run perf`). Embedded or background browser tabs throttle `requestAnimationFrame`, so measure in a real foreground window.
- Tiny networks are **latency-bound, not compute-bound**: on a GPU each step costs a few kernel launches, so a 2-8-8-1 net is not faster on WebGPU than on CPU. The GPU pays off on MNIST.

## Not built (stretch goals)

Tiny transformer mode, experiment side-by-side comparison, loss landscape, ONNX import and the hand-written WebGPU kernel benchmark are **not implemented**.
The `Trainer` interface and the snapshot/protocol design leave room for them.
