# Interview questions about LearnScope

Answers below are **drafts**: read the code they point at, then rewrite each in your own words. An interviewer can tell the difference.
Numbers quoted (236 KB, ~4.7 s/epoch, 5,386 parameters) were measured on the developer's machine; re-measure on yours.

## Backpropagation and calculus

**1. Explain backpropagation in one minute.**
Forward pass computes the loss. Then, starting from `∂L/∂L = 1`, we go through the graph in reverse; each node multiplies the gradient arriving from its consumers by its own local derivative and passes it to its inputs. That is the chain rule, organised so each intermediate gradient is computed once. (`src/engine/tiny/value.ts`)

**2. Why do gradients accumulate with `+=`?**
A value used in several places affects the loss through several paths; the total derivative is the sum over paths.

**3. What is a computational graph and how do you traverse it?**
Nodes are values, edges are the operations that produced them. We do a depth-first topological sort so every node comes after its parents, then iterate that list in reverse.

**4. How did you verify your gradients?**
Central differences: `(f(x+h) − f(x−h)) / 2h` with `h = 1e-6`, compared to the analytic gradient; all agree within 1e-6 for each op and for a whole MLP. (`value.test.ts`, `cpu/mlp.test.ts`)

**5. What's the derivative of tanh, sigmoid and ReLU?**
`1 − tanh²`, `σ(1 − σ)`, and `1 if x>0 else 0`.

**6. For squared error with a linear output, what is `∂L/∂ŷ`?**
`2(ŷ − t)`. (For sigmoid + binary cross-entropy it simplifies to `p − y`, which is what the CPU MLP uses.)

**7. What is the difference between a derivative, a gradient and a Jacobian?**
Derivative: 1 input → 1 output slope. Gradient: vector of partial derivatives of a scalar loss. Jacobian: matrix of partials for vector → vector functions; backprop multiplies Jacobian-vector products without forming full Jacobians.

**8. Why is backprop efficient compared with numerical differentiation?**
One backward pass gives gradients for _all_ parameters at about the cost of one forward pass; numerical differentiation needs 2 forward passes per parameter.

## Optimisers, learning rate, regularisation

**9. SGD vs momentum vs Adam?**
SGD: step against the gradient. Momentum: step along a running average of gradients, smoothing noise and speeding up consistent directions. Adam: additionally divides by the running RMS of gradients so each parameter gets its own scale. (`trainer.ts → applyUpdate`)

**10. Why does Adam have bias correction?**
`m` and `v` start at 0, so early averages are biased toward 0; dividing by `1 − β^t` fixes that.

**11. How does the Model Doctor decide the learning rate is too high?**
Over the last 20 snapshots, either ≥30% of steps raise train loss by >5%, or the recent average is >1.3× the best loss seen. Suggested fix: lr/3 capped at a stable ceiling for the optimizer (Adam 0.01, SGD 0.1), with a restart because a network that overshot may be stuck in saturated or dead regions. (`rules/lrHigh.ts`)

**12. And too low?**
After ≥150 steps, the last 30 snapshots (≥60 steps) improved loss by <1% while the loss is still high (>0.45) and accuracy <95%.

**13. What is L2 regularisation and how did you implement it?**
Add `½·λ·Σw²` to the loss, which adds `λw` to each weight's gradient (weight decay); it penalises large weights, smoothing the function. It is a live-adjustable setting.

**14. Batch size trade-offs?**
Small: noisy gradients (regularising), more updates per epoch, poor hardware use. Large: smooth gradients, better GPU use, fewer updates per epoch.

**15. What is an epoch, and how do you compute it?**
One pass over the training set: `step × batchSize / nTrain`.

## Activations, initialisation, training problems

**16. Why do we need non-linear activations?**
Stacked linear maps collapse into a single linear map. Non-linearity lets deeper nets represent curved boundaries like the spiral.

**17. Vanishing gradients: cause and fix?**
Each layer multiplies the gradient by the activation's slope (≤1 for tanh, ≤0.25 for sigmoid), so early layers get tiny gradients. Fixes: ReLU/GELU, better init, residual connections, normalisation. The Doctor compares first-layer vs last-layer gradient norms and suggests ReLU.

**18. Exploding gradients?**
Weights/lr large enough that updates grow each step; gradient norms or loss blow up to inf/NaN. Fix: lower lr, gradient clipping, better init.

**19. What is a dead ReLU?**
A unit whose pre-activation is negative for every input, so output and gradient are zero and it never updates. The Doctor measures the fraction of units whose max activation over the training set is ~0 and flags >50%.

**20. Why GELU as a suggested fix for dead ReLUs?**
GELU is smooth and has a non-zero gradient for small negative inputs, so units can recover.

**21. He vs Glorot initialisation?**
He: `std = √(2/nin)` suits ReLU (half the units are zeroed). Glorot: `√(2/(nin+nout))` keeps variance for tanh/sigmoid. Both keep activation and gradient scales stable across layers.

**22. What is saturation and why did one rule give a false alarm?**
tanh/sigmoid outputs pinned at the flat ends have ~0 slope. But a converged spiral network legitimately saturates; the rule now only fires while loss is still high. Found by manual testing, locked by a unit test.

**23. How do you detect overfitting from telemetry?**
Test loss minus train loss above 0.15, accuracy gap above 5%, and test loss risen more than 10% from its minimum. Evidence jumps to the step where test loss was lowest.

**24. Why isn't loss alone enough to debug a model?**
It doesn't say where or why training fails. Per-layer gradient, activation and weight statistics localise the problem.

## CNNs and MNIST

**25. What does a convolution do?**
Slides a small filter over the input computing a dot product at each position, producing a feature map showing where a pattern occurs. Weights are shared across positions.

**26. Why are CNNs better for images than MLPs?**
Weight sharing (fewer parameters), locality (nearby pixels matter together), and translation equivariance. Here: 5,386 CNN parameters vs ~109k for the MLP at similar or better accuracy.

**27. What does max-pooling do?**
Takes the max in each window, shrinking the map and giving small-shift tolerance.

**28. How are the CNN's output sizes computed?**
28×28 →(conv 5×5, valid) 24×24×8 →(pool) 12×12×8 →(conv 3×3) 10×10×16 →(pool) 5×5×16 → flatten 400 → dense 10.

**29. How do you read a confusion matrix?**
Rows are the true class, columns the prediction; the diagonal is correct; bright off-diagonal cells show systematic confusions (4↔9, 3↔5).

**30. How does the draw-a-digit pad make drawings look like MNIST?**
Scale to a 20×20 box inside a 28×28 frame, then shift so the centre of mass is at the centre (as MNIST was prepared). Without this, predictions are poor. (`canvasToMnist`)

**31. Why a subset of MNIST?**
10k train + 2k test is enough to show learning, ships as a 2.3 MB PNG, trains an epoch in seconds in the browser, and avoids a runtime dependency on another host.

**32. What happens if you train longer on a small subset?**
Train accuracy approaches 100% while test accuracy plateaus around the high 90s: the beginning of overfitting.

## WebGPU, Web Workers, performance

**33. Why train in a Web Worker?**
Training, GPU submissions and data readback would block the main thread and make the UI stutter. The worker owns the model; the UI only receives snapshots.

**34. How do you move data between worker and UI efficiently?**
`postMessage` with **transferable** `ArrayBuffer`s (zero-copy). Snapshots are small and batched per animation frame.

**35. WebGPU vs WebGL for ML?**
WebGL repurposes graphics pipelines (textures/fragment shaders). WebGPU has compute shaders, explicit buffers and lower overhead, giving better ML performance. TF.js picks WebGPU, then WebGL, then CPU. (`engine/tf/backend.ts`)

**36. What is the performance of the MNIST CNN?**
About 2,100 images/s (~4.7 s per 10k-image epoch) on the developer's WebGPU laptop; the page shows the live number. Budget was <60 s.

**37. Is a GPU faster for the playground's tiny networks?**
Not necessarily: tiny matrices are latency-bound by kernel launches and readbacks. The GPU pays off on MNIST/CNNs.

**38. How did you keep the first load small?**
Code-splitting: the first-load chunk is 236 KB (75 KB gzip); TF.js (~2 MB) is a lazy worker chunk. The landing animation uses a dependency-free CPU MLP, so the home page never loads TensorFlow.js.

**39. How do you avoid memory leaks with TF.js?**
Tensors on the GPU aren't garbage collected: use `tf.tidy`, `tf.keep` for state that must survive, and `dispose()`. A unit test checks the tensor count doesn't grow across steps.

**40. Why Canvas instead of SVG/DOM for charts?**
Thousands of points redraw every frame; canvas avoids creating DOM nodes. SVG is used only for the small static backprop diagram.

## Design and engineering decisions

**41. How does time travel work?**
Each snapshot stores weights and per-layer statistics. Every view is a pure function of "the selected snapshot", so scrubbing re-renders consistently. The decision boundary is recomputed from stored weights.

**42. How do you bound memory for the history?**
A ring buffer with item and byte limits that thins the oldest half (keeps every second item) rather than discarding the start of training.

**43. Why compute the decision boundary on the main thread?**
Tiny nets make a 48×48 forward pass cheap, snapshots stay small, and time-travel is exact for any recorded step. The CPU forward is unit-tested against TF.js.

**44. How do you prevent race conditions between reconfiguring and training?**
The worker serialises messages through one promise queue, stops the loop and awaits it before rebuilding, and emits a `built` message that tells the UI to discard older snapshots.

**45. How are shareable links implemented and secured?**
The config is JSON → base64url in the URL hash. On load it is **sanitised**: unknown values fall back to defaults and numbers are clamped, so a malicious link can't crash the app or request absurd sizes. (`urlConfig.ts`, tested)

**46. How do you test ML code deterministically?**
Seeded RNG for data and initialisation; numerical gradient checks; independent reference implementations (CPU forward vs TF.js); synthetic telemetry histories for the Doctor instead of real training runs.

**47. What would you do next?**
Experiment comparison, a tiny transformer with attention visualisation, a loss-landscape slice, and hand-written WebGPU kernels for dense forward/backward benchmarked against TF.js.

**48. What was the hardest bug?**
Pick your own from this list and say what you did: (a) Vite dev server failing because the project path contained `#`; (b) a false-positive Doctor rule on a healthy spiral run; (c) double-disposing shared layers in the MNIST feature-map model.

**49. How is accessibility handled?**
Labelled controls, `aria-label`s on canvases, an "ⓘ What am I looking at?" on every chart, keyboard shortcuts that skip form fields, `prefers-reduced-motion` respected, and a visible notice when WebGPU is unavailable.

**50. What are the limits of the Model Doctor?**
It is heuristic: thresholds were tuned on a handful of runs, so it can miss or misjudge unusual setups. Being rule-based makes it explainable and unit-testable, and each rule documents its evidence.
