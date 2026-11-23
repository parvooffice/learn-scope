# Resume bullets (draft: edit, and keep only numbers you can defend)

- Built **LearnScope**, a Python web app that trains neural networks live and visualises every weight, activation and gradient (Python, NumPy, NiceGUI, Plotly); the whole engine (autograd, backprop, SGD / momentum / Adam, a convolutional network) is hand-written with no deep-learning framework, and a live UI stays attached to a training thread at about **86 steps/s**.
- Implemented a **from-scratch autograd engine** (computational graph, topological backward pass) with gradients verified against numerical differentiation to **<1e-6**, powering an interactive step-by-step backpropagation explainer.
- Designed a rule-based **"Model Doctor"** that detects vanishing/exploding gradients, dead ReLUs, saturation, bad learning rates and overfitting from per-layer training telemetry, with evidence links and one-click fixes; covered by deterministic unit tests on synthetic histories.
- Added **time travel** (bounded snapshot history with memory-aware thinning, 150 MB cap) so every view re-renders exactly as the network was at any recorded step.
- Wrote a **CNN in NumPy** (im2col convolution, max-pool, softmax cross-entropy, Adam; 5,386 parameters) reaching **96.9% test accuracy** on a 10k-image MNIST subset at ~1,350 img/s on a laptop CPU, with filter / feature-map views, a confusion matrix and a draw-a-digit pad.
- Found and fixed a performance bug by measuring: building validated Plotly figures starved the training thread (25 steps/s); switching to plain dict figures and in-place updates gave a ~3.4× speed-up, now guarded by a test.
- Tooling: **114 pytest tests**, ruff, GitHub Actions CI on three Python versions, Dockerfile.
