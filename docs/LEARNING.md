# Learning log

One section per milestone: the concepts (plain language), a self-quiz, and "how I'd explain this in an interview".
**Try to answer each quiz question yourself before opening the answer.** Where a section says "Your turn", the exercise is in
[EXERCISES.md](EXERCISES.md).

---

## 1. Foundations: what is training?

**Concepts**

- A **tensor** is just a grid of numbers: a scalar (0-D), vector (1-D), matrix (2-D), or a stack of matrices (3-D+). An image is a 28×28 tensor; a batch of 64 images is 64×28×28.
- A **model** is a function with adjustable numbers (**parameters** / weights). A **loss** measures how wrong its outputs are. **Training** is a loop:
  1. forward pass: compute the model's output on some data,
  2. compute the loss,
  3. backward pass: compute the **gradient** (how the loss changes if each parameter changes a little),
  4. update: nudge every parameter against its gradient.
     Repeat thousands of times.
- Where in the code: `src/engine/tf/trainer.ts` → `trainStep()` is exactly this loop.

**Quiz**

1. What are the four steps of a training loop?
2. Why do we call a batch of images a 3-D tensor?
3. What does the loss number tell you, and what doesn't it tell you?

<details><summary>Answers</summary>

1. Forward, loss, backward (gradients), update.
2. batch × height × width: one index for which image, two for the pixel position (a colour image would add a 4th: channels).
3. It says how far predictions are from the targets on this data. It does not say _why_ it is high (bad learning rate? dead neurons? wrong model?), which is why LearnScope shows internals.

</details>

**In an interview:** "Training is repeated forward pass, loss, backward pass and update; I built both a from-scratch version and a TensorFlow.js one so I can show the same loop at two levels."

---

## 2. The tiny autograd engine

**Concepts**

- A **derivative** is the slope: how much the output changes per tiny change of the input.
- The **chain rule**: if `y = f(g(x))`, then `dy/dx = f'(g(x)) · g'(x)`. Slopes multiply along a chain.
- A **computational graph**: every `+`, `×`, `tanh` creates a node that remembers its parents. `backward()` visits nodes in reverse (topological) order and applies each node's _local_ derivative times the gradient flowing in. If a value is used twice its gradients **add up** (`+=`).
- Where: `src/engine/tiny/value.ts`. Tests prove the gradients match numerical differentiation to < 1e-6 (`value.test.ts`).

**Your turn:** write `backward` for `mul` and `tanh` (EXERCISES.md #1).

**Quiz**

1. For `out = a × b`, what are `d out/da` and `d out/db`?
2. Why does `backward()` need a topological order?
3. Why do we write `this.grad += ...` and not `=`?
4. How do we know our gradients are right?

<details><summary>Answers</summary>

1. `b` and `a`.
2. A node's gradient is complete only after _all_ nodes that use it have sent theirs; reverse topological order guarantees that.
3. A value can feed several operations; the total derivative is the sum of the contributions (multivariable chain rule).
4. Numerical gradient check: `(f(x+h) − f(x−h)) / 2h` must match the analytic gradient (`value.test.ts`).

</details>

**In an interview:** "Each op stores a closure with its local derivative; backward walks the graph in reverse topological order accumulating `grad += local × upstream`. I verified it against central differences within 1e-6."

---

## 3. Backprop explainer (`/#/backprop`)

**Concepts**

- For `z = w·x + b`, `h = tanh(z)`, `ŷ = Σ v·h + c`, `L = (ŷ − t)²`: `∂L/∂ŷ = 2(ŷ − t)`, then `∂L/∂v = ∂L/∂ŷ · h`, `∂L/∂h = ∂L/∂ŷ · v`, `∂L/∂z = ∂L/∂h · (1 − h²)`, `∂L/∂w = ∂L/∂z · x`.
- Each line is "gradient arriving × local slope". **Backprop is the chain rule organised so every intermediate gradient is computed once.**
- `1 − h²` is tiny when `|h|` is near 1: this is the **vanishing-gradient** mechanism for tanh.
- The page shows both the hand-written chain rule and the autograd engine and the difference between them (~0).

**Quiz**

1. Why is the update `p ← p − lr · ∂L/∂p` (minus)?
2. What happens to `∂L/∂z` when `h ≈ ±1`?
3. What changes if the learning rate is 1.5 in the explainer? Try it.

<details><summary>Answers</summary>

1. The gradient points uphill (direction of increasing loss); we want to go downhill.
2. `1 − h² ≈ 0`, so the gradient is squashed to near zero: that neuron barely learns (saturation).
3. The step overshoots; loss bounces or grows (same story as "learning rate too high" in the Doctor).

</details>

**In an interview:** "I can walk through backprop on a 2-3-1 network with numbers: forward values, the loss, then ∂L/∂ŷ, through the output weights, through tanh's slope, down to w; and show the update lowers the loss."

---

## 4. Playground: batches, epochs, optimizers, learning rate

**Concepts**

- **Batch**: the number of samples used per step. Small batches are noisy but cheap; large ones are smooth. **Epoch**: one pass over the whole training set (`epoch = step × batch / N`).
- **SGD**: `p -= lr·g`. **Momentum**: keeps a running average of gradients (`m = 0.9m + g`), so steps smooth out and speed up along consistent directions. **Adam**: also divides by a running average of squared gradients, giving every parameter its own step size.
- **Learning rate (lr)**: too small, slow; too large, overshoot and diverge.
- **Train/test split**: test points are never trained on; they measure generalisation.
- Why the worker? TensorFlow.js work and per-step readbacks would stutter the UI on the main thread.
- Where: `src/engine/tf/trainer.ts` (optimizers are ~30 lines, in `applyUpdate`), `src/worker/training.worker.ts`.

**Quiz**

1. If you double the batch size, what happens to the number of steps per epoch?
2. Why does Adam often work with a learning rate that makes plain SGD diverge?
3. Why must the test set be held out?

<details><summary>Answers</summary>

1. It halves.
2. Adam normalises each parameter's step by the running RMS of its gradient, so the effective step is bounded by roughly `lr` regardless of gradient scale.
3. If the model trains on it, test accuracy no longer measures performance on unseen data.

</details>

---

## 5. Network graph and internals: activations, initialisation, vanishing gradients

**Concepts**

- **Activation functions** add curvature. Without them, any stack of layers is just one linear map. ReLU: `max(0,x)`. tanh/sigmoid: smooth S-curves whose slope goes to 0 at the ends. GELU: a smooth ReLU-like curve.
- **Initialisation**: random but scaled. He init `√(2/nin)` for ReLU/GELU; Glorot `√(2/(nin+nout))` for tanh/sigmoid, so activations and gradients keep a similar size layer to layer.
- **Vanishing gradients**: each layer multiplies the gradient by a slope (≤ 1 for tanh, ≤ 0.25 for sigmoid), so early layers get almost nothing in deep nets. **Dead ReLU**: a unit that outputs 0 for every input has zero gradient and can never recover.
- The weight/gradient/activation histograms and the "dead %" labels let you _see_ all of this. A dead unit is drawn dashed with × in the network graph.

**Quiz**

1. Why is the max slope of sigmoid a problem in deep networks?
2. How does a ReLU unit "die"?
3. Why not initialise all weights to zero?

<details><summary>Answers</summary>

1. 0.25 per layer, so after 5 layers the gradient is shrunk by ~1000×.
2. A big update pushes its weights so its pre-activation is negative for all data; output and gradient are 0 forever.
3. All neurons in a layer would compute the same thing and receive identical gradients: symmetry is never broken.

</details>

---

## 6. Time travel

**Concepts**

- Every few steps the worker records a **snapshot** (weights, per-layer gradient/activation stats, metrics). They are stored in a **bounded history buffer**; when memory or item limits are reached we thin out _older_ snapshots instead of dropping them.
- Every view is a pure function of "the selected snapshot": that is why scrubbing redraws everything consistently. The decision boundary is recomputed on the main thread from that snapshot's weights.
- Design lesson: keep views as pure functions of immutable data and time travel comes almost for free.

**Quiz**

1. Why thin the oldest half instead of deleting the oldest snapshot?
2. Why compute the boundary from stored weights rather than store the boundary image?

<details><summary>Answers</summary>

1. We keep the ability to rewind to the beginning; older history just gets coarser.
2. Weights are tiny and let us redraw at any resolution/theme; images for every snapshot would use far more memory.

</details>

---

## 7. Model Doctor

**Concepts**

- Rule-based diagnosis: each rule is a pure, deterministic function of the telemetry history with named thresholds; each returns severity, evidence (step + chart), an explanation and a fix.
- Rules are **unit-tested with synthetic histories**: build a fake history that should trigger a rule, and one that should not. This avoids slow flaky "train a real net" tests.
- A real bug found by manual testing: the saturation rule flagged a _healthy_ converged spiral net. The fix (only flag while loss is high) is now a test.

**Your turn:** write a rule end to end (EXERCISES.md #3).

**Quiz**

1. Why use a full-batch gradient for the vanishing-gradient rule?
2. Why does the Doctor hide other diagnoses when the run has exploded?
3. How would you reduce false positives in a rule?

<details><summary>Answers</summary>

1. Mini-batch gradients are noisy; the ratio between layers needs a stable measurement.
2. They are consequences (NaN everywhere), not causes. Show the root cause.
3. Add guard conditions (e.g. loss still high, enough history), tune thresholds on real runs, and add negative test cases.

</details>

---

## 8. MNIST: convolutions, pooling, overfitting

**Concepts**

- A **convolution** slides a small filter (here 5×5) over the image and computes a dot product at each position, producing a **feature map** that lights up where the pattern occurs. The _same_ weights are reused everywhere: far fewer parameters than a dense layer and built-in translation equivariance.
- **Max-pooling** keeps the strongest response in each 2×2 block: shrinks the map and adds tolerance to small shifts.
- The small CNN here has 5,386 parameters; the MLP has ~109k. The CNN usually matches or beats it.
- **Overfitting**: train accuracy keeps rising while test accuracy stalls. Fight it with more data, smaller models, L2, dropout, early stopping.
- The confusion matrix shows _which_ digits get mixed up (4↔9, 3↔5, 7↔2).

**Quiz**

1. Why does a conv layer need fewer parameters than a dense layer on 28×28 images?
2. What does max-pooling do to the feature map size and why does it help?
3. In the confusion matrix, where do you look for the model's mistakes?

<details><summary>Answers</summary>

1. Weight sharing: one 5×5 filter (25 weights) is applied at every position instead of a separate weight per pixel per neuron.
2. Halves width and height; keeps the strongest activation so small shifts don't change the result, and reduces compute.
3. Off-diagonal cells: row = true digit, column = predicted digit.

</details>

---

## 9. Polish and performance

**Concepts**

- **Code-splitting** (`React.lazy` + dynamic `import()`) keeps the first load at 236 KB; TF.js (~2 MB) only loads when needed.
- **Canvas** is much cheaper than thousands of DOM/SVG nodes for live-updating charts.
- **Measure, don't guess:** the MNIST page reports live images/second.
- Accessibility: every control has a label, canvases have `aria-label`s, keyboard shortcuts skip form fields, reduced-motion is respected.

**Quiz**

1. Why batch UI updates to one per animation frame?
2. Why is a tiny network not necessarily faster on a GPU?

<details><summary>Answers</summary>

1. More frequent updates can't be shown anyway and would waste CPU on React renders.
2. GPU work has fixed launch/transfer overhead; tiny matrices don't have enough parallel work to hide it.

</details>
