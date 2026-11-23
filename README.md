# LearnScope — watch a neural network learn, live

LearnScope trains neural networks in front of you and shows everything while it happens: every weight, activation and
gradient. You can pause, **rewind to any training step**, and let a rule-based **Model Doctor** diagnose what is going
wrong and suggest a fix.

It is written in **Python**. The engine (autograd, backprop, optimizers, a convolutional network, the Doctor) is
hand-written with **NumPy**, with no PyTorch or TensorFlow. The web UI is served from Python by **NiceGUI** with
**Plotly** charts.

- **GIF / video:** _record the 2-minute script in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) and drop a GIF here_

## What is in it

| Page               | What it does                                                                                                                                                                                                                                  |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Playground**     | 5 toy datasets, a model builder (layers, neurons, activations, optimizer, LR, batch, L2), live decision boundary, network graph (click a neuron to see what it detects), loss/accuracy/gradient/weight/activation charts, time travel.          |
| **Model Doctor**   | 7 deterministic, unit-tested rules: exploding & vanishing gradients, dead ReLUs, saturated units, learning rate too high / too low, overfitting. Each gives severity, evidence (jumps to the step), explanation and a one-click fix.            |
| **Compare**        | Two runs on the **same data and seed**, side by side: overlaid loss/accuracy curves, two decision boundaries and a live "who is ahead" verdict. Defaults to Adam vs SGD; change anything on either side.                                      |
| **Loss landscape** | In the Playground: a 2-D slice of the loss surface around the weights at the step you are viewing, with the training path drawn on it.                                                                                                        |
| **Backprop**       | A guided 10-step walk through forward pass → loss → chain rule backward → weight update on a 2-3-1 network, every number computed live by the from-scratch autograd engine and cross-checked against a hand-written chain rule.                |
| **MNIST**          | A small CNN (5,386 parameters) or an MLP (109,386) trained on 10k real digits in NumPy: filters, feature maps, confusion matrix, most confident mistakes and a draw-a-digit pad.                                                              |
| **Share**          | The full experiment config is encoded in the URL (`/playground?c=…`) and sanitised on load.                                                                                                                                                   |

## Run it locally

Requires Python 3.10+ (developed on 3.14).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
python -m learnscope            # http://localhost:8080
```

Settings through environment variables: `PORT` (default 8080), `HOST` (default `127.0.0.1`, set `0.0.0.0` to serve a
network) and `LEARNSCOPE_SECRET` (session secret; random per start if unset).

```bash
python -m pytest                # the whole test suite (114 tests, about 15 s)
```

### Docker

```bash
docker build -t learnscope .
docker run -p 8080:8080 learnscope
```

### Hosting

LearnScope is a stateful Python server (training runs on background threads), so it needs a host that runs a Python
process, not a static-file host. Any of Render, Railway, Fly.io or a small VM works: start command `python -m learnscope`
with `HOST=0.0.0.0` and `PORT` set by the platform (or use the Dockerfile). Every open browser tab runs its own
training thread, so size the machine for the number of simultaneous viewers.

## Architecture in one picture

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

Details and the reasoning behind each decision: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Measured numbers (honest, on one machine)

Measured on the developer's Windows laptop (CPU only, Python 3.14, NumPy). Your numbers will differ.

| What                                              | Measured                                                                                               |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Playground step, default spiral (2×8 tanh, Adam)  | about 0.15 ms per training step; a snapshot costs about 1.1 ms                                        |
| Playground with the live UI attached              | about 86 training steps per second (the UI thread and the training thread share the GIL)              |
| MNIST small CNN (5,386 parameters)                | about 1,350 images/s; 92.4% test accuracy after 1 epoch, 96.9% after about 10 epochs (~75 s)          |
| MNIST MLP 784-128-64-10 (109,386 parameters)      | about 4,700 images/s; 91.3% after 1 epoch, 93.6% after about 10 (it starts to overfit the 10k subset) |
| Snapshot memory                                   | hard limit 150 MB; the history thins its oldest half instead of dropping the start of training        |

A UI refresh set (all charts of a playground tick) is guarded by a test at under 25 ms.

**Seed sensitivity.** A 2-layer tanh network on a spiral is sensitive to its random initialisation: the default seed (3)
learns reliably, while for example seed 7 sits on a plateau (loss about 0.66) for thousands of steps. This is
real behaviour of the model, not a bug, and exactly the kind of thing the Doctor and the Compare page are for.

## Tests

`python -m pytest` runs 114 tests:

- the scalar autograd engine against analytic derivatives and central differences (< 1e-6);
- the NumPy MLP gradients against finite differences, optimizers, activations and the engine's determinism;
- every Model Doctor rule on synthetic histories (trigger, no false alarm, too-short history), plus a regression test for the starvation bug described in the architecture notes;
- the MNIST layers (convolution, max-pool, dense, softmax-CE) against numerical gradients;
- loss-landscape directions and projection, the run thread (play, pause, time travel, reconfigure), config sanitising and URL round-trips, the drawing pad, and the chart builders (strict JSON, speed guard).

## What I learned / how it was built

Built with AI assistance. The learning material is in the repo so every part can be explained, not just run:

- [docs/LEARNING.md](docs/LEARNING.md): concepts per milestone, with self-quiz questions and answers.
- [docs/EXERCISES.md](docs/EXERCISES.md): the pieces worth writing yourself (backward of `mul` and `tanh`, a dense layer, a Doctor rule), with tests.
- [docs/INTERVIEW.md](docs/INTERVIEW.md): 50+ interview questions about this project with draft answers.
- [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md): the 2-minute demo.

## History

The first version of LearnScope was a TypeScript app (React + TensorFlow.js on WebGPU) and still lives on the `main`
branch. This `python` branch is a rewrite of the whole project in Python: same ideas, new engine, server-side
training. Trade-offs of the move: no in-browser GPU, a server process instead of a static site, and a lower redraw rate
(a few updates per second over a websocket instead of 60 fps), in exchange for one language end to end, a NumPy
engine that is easy to read and test, and a real convolution/backprop implementation instead of library calls.
