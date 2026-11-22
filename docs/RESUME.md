# Resume bullets (draft: edit, and keep only numbers you can defend)

- Built **LearnScope**, an in-browser deep-learning visualizer that trains models on the GPU (TensorFlow.js + WebGPU, WebGL/CPU fallback) in a Web Worker and renders weights, activations and gradients live; measured **60 fps** UI (p95 frame 16.9 ms, no long tasks) while training, with a **236 KB** first-load bundle.
- Implemented a **from-scratch autograd engine** (computational graph, topological backward pass) with gradients verified against numerical differentiation to **<1e-6**, powering an interactive step-by-step backpropagation explainer.
- Designed a rule-based **"Model Doctor"** that detects vanishing/exploding gradients, dead ReLUs, saturation, bad learning rates and overfitting from per-layer training telemetry, with evidence links and one-click fixes; **29 deterministic unit tests** on synthetic histories.
- Added **time travel** (bounded snapshot history with memory-aware thinning) so every view re-renders exactly as the network was at any recorded step.
- Trained an MLP/CNN on **MNIST in the browser** (~2,100 img/s, ~4.7 s per 10k-image epoch on WebGPU on my laptop) with filter/feature-map visualisation, confusion matrix and a draw-a-digit pad.
- Tooling: TypeScript (strict), Vitest (74 tests), Playwright e2e, ESLint/Prettier, GitHub Actions CI.
