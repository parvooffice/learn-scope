# LearnScope — watch a neural network learn, live

LearnScope trains neural networks **in your browser** (TensorFlow.js on WebGPU, with WebGL/CPU fallback) and
shows everything while it happens: every weight, activation and gradient. You can pause, **rewind to any training
step**, and let a rule-based **Model Doctor** diagnose what is going wrong and suggest a fix.

- **Live demo:** _add your Vercel URL here after deploying (see "Deploy" below)_
- **GIF / video:** _record the 2-minute script in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) and drop a GIF here_

## What is in it

| Page               | What it does                                                                                                                                                                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Playground**     | 5 toy datasets, a model builder (layers, neurons, activations, optimizer, LR, batch, L2), live decision boundary, network graph (click a neuron to see what it detects), loss/accuracy/gradient/weight/activation charts, time travel.                             |
| **Model Doctor**   | 7 deterministic, unit-tested rules: exploding & vanishing gradients, dead ReLUs, saturated units, learning rate too high / too low, overfitting. Each gives severity, evidence (jumps to the step), explanation and a one-click fix.                               |
| **Compare**        | Two experiments train side by side on the **same data**, each in its own worker: overlaid loss/accuracy curves, a scoreboard, two boundaries and graphs, and a Doctor per run. Five presets (Adam vs SGD, ReLU vs tanh, huge learning rate, small vs big net, L2). |
| **Loss landscape** | In the Playground: a 2-D slice of the loss surface around the weights at the step you are viewing, with the training path drawn on it.                                                                                                                             |
| **Backprop**       | A guided 10-step walk through forward pass → loss → chain rule backward → weight update on a 2-3-1 network, every number computed live by the from-scratch autograd engine.                                                                                        |
| **MNIST**          | MLP or small CNN trained on 10k real digits on the GPU: filters, feature maps, confusion matrix, most-confused examples and a draw-a-digit pad.                                                                                                                    |
| **Share**          | The full experiment config is encoded in the URL (`#/playground?c=…`).                                                                                                                                                                                             |

## Run it locally

Requires Node 20+.

```bash
npm install
npm run dev        # http://localhost:5173
```

Other scripts:

```bash
npm test           # unit tests (Vitest)
npm run lint       # ESLint
npm run typecheck  # tsc
npm run build      # production build into dist/
npm run preview    # serve dist/ at http://localhost:4173
npm run perf       # measure FPS / long tasks while training (needs `npm run preview` running)
npm run e2e        # Playwright end-to-end tests (uses installed Chrome; PW_CHANNEL=chromium for the bundled one)
```

> **Windows note:** keep the project in a folder path **without a `#` character**. Vite's dev server cannot
> serve files from such a path (this project was moved for exactly that reason).

## Deploy (free, static)

The app is a static site: no backend.

- **Vercel:** import the repo, framework preset _Vite_, build command `npm run build`, output `dist`. `vercel.json` is included.
- **Anything else:** upload the contents of `dist/` (Netlify, GitHub Pages, S3…). Routing uses the URL hash, so no server rewrites are needed.
- **Download and open:** `npm run build && npm run preview`.

## Architecture in one picture

```
 React UI (main thread)                       Web Worker (training.worker.ts)
 ┌──────────────────────────┐   commands      ┌──────────────────────────────┐
 │ Zustand stores           │ ──────────────▶ │ TensorFlow.js (WebGPU/WebGL) │
 │ Canvas views             │                 │ hand-written training loop   │
 │ Model Doctor (pure fns)  │ ◀────────────── │ per-layer statistics         │
 │ history ring buffer      │  snapshots      └──────────────────────────────┘
 └──────────────────────────┘ (transferable)
```

Details and the reasoning behind each decision: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Measured numbers (honest, on one machine)

Measured on the developer's Windows laptop with WebGPU in Chrome/Edge. Your numbers will differ; the app shows live throughput on the MNIST page.

| Budget                                       | Target   | Measured                                                                                                      |
| -------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------- |
| First-load JavaScript (before TensorFlow.js) | < 300 KB | **236 KB** raw / **75 KB** gzip (`index` chunk)                                                               |
| MNIST small CNN, 1 epoch on 10k images       | < ~60 s  | **~4.7 s** (≈2,100 img/s, WebGPU)                                                                             |
| Snapshot memory                              | < 150 MB | hard limit 150 MB; ~3 KB per snapshot (observed: 422 snapshots ≈ 1.2 MB)                                      |
| 60 fps UI while training                     | 60 fps   | **60 fps**, p95 frame 16.9 ms, 0 long tasks (headless Chrome + WebGPU, default spiral config; `npm run perf`) |

## Tests

- `npm test`: unit tests for the autograd engine (gradients vs numerical differentiation < 1e-6), the CPU MLP gradients, every Model Doctor rule (synthetic histories),
  dataset generators, URL encoding, the history ring buffer, the TF.js trainers (checked against an independent CPU forward pass) and the backprop explainer.
- `npm run e2e`: opens the playground, trains, checks the loss falls and the decision boundary changes, then scrubs the timeline back.

## What I learned / how it was built

Built with AI assistance. The learning material is in the repo so every part can be explained, not just run:

- [docs/LEARNING.md](docs/LEARNING.md): concepts per milestone, with self-quiz questions and answers.
- [docs/EXERCISES.md](docs/EXERCISES.md): the pieces worth writing yourself (backward of `mul` and `tanh`, a dense layer, a Doctor rule), with tests.
- [docs/INTERVIEW.md](docs/INTERVIEW.md): 40+ interview questions about this project with draft answers.
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md): the 2-minute demo.

## Browser support

Current Chrome and Edge (WebGPU). Firefox and Safari fall back to WebGL or CPU with a visible notice.
