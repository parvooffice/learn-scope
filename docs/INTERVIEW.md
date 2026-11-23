# Interview questions about LearnScope

Answers below are **drafts**: read the code they point at, then rewrite each in your own words. An interviewer can tell the difference.
Numbers quoted (5,386 parameters, ~1,350 img/s, 96.9% test accuracy, ~86 steps/s) were measured on the developer's machine; re-measure on yours.

## Backpropagation and calculus

**1. Explain backpropagation in one minute.**
Forward pass computes the loss. Then, starting from `dL/dL = 1`, we go through the graph in reverse; each node multiplies the gradient arriving from its consumers by its own local derivative and passes it to its inputs. That is the chain rule, organised so each intermediate gradient is computed once. (`learnscope/autograd.py`)

**2. Why do gradients accumulate with `+=`?**
A value used in several places affects the loss through several paths; the total derivative is the sum over paths.

**3. What is a computational graph and how do you traverse it?**
Nodes are values, edges are the operations that produced them. We do a depth-first topological sort (iteratively, so a deep graph cannot hit Python's recursion limit) so every node comes after its parents, then iterate that list in reverse.

**4. How did you verify your gradients?**
Central differences: `(f(x+h) - f(x-h)) / 2h` with `h = 1e-6`, compared to the analytic gradient; all agree within 1e-6 for each op and for a whole MLP. The NumPy MLP and every MNIST layer are checked the same way. (`tests/test_autograd.py`, `tests/test_engine.py`, `tests/test_mnist.py`)

**5. What's the derivative of tanh, sigmoid and ReLU?**
`1 - tanh²`, `σ(1 - σ)`, and `1 if x > 0 else 0`.

**6. For squared error with a linear output, what is `dL/dŷ`?**
`2(ŷ - t)`. (For sigmoid + binary cross-entropy it simplifies to `p - y`, which is what the NumPy MLP uses.)

**7. What is the difference between a derivative, a gradient and a Jacobian?**
Derivative: 1 input → 1 output slope. Gradient: vector of partial derivatives of a scalar loss. Jacobian: matrix of partials for vector → vector functions; backprop multiplies Jacobian-vector products without forming full Jacobians.

**8. Why is backprop efficient compared with numerical differentiation?**
One backward pass gives gradients for _all_ parameters at about the cost of one forward pass; numerical differentiation needs 2 forward passes per parameter.

## Optimisers, learning rate, regularisation

**9. SGD vs momentum vs Adam?**
SGD: step against the gradient. Momentum: step along a running average of gradients, smoothing noise and speeding up consistent directions. Adam: additionally divides by the running RMS of gradients so each parameter gets its own scale. (`learnscope/mlp.py → Optimizer`)

**10. Why does Adam have bias correction?**
`m` and `v` start at 0, so early averages are biased toward 0; dividing by `1 - β^t` fixes that.

**11. How does the Model Doctor decide the learning rate is too high?**
Over the last 20 snapshots, either ≥30% of steps raise train loss by >5%, or the recent average is >1.3× the best loss seen. Suggested fix: lr/3 capped at a stable ceiling for the optimizer (Adam 0.01, momentum 0.05, SGD 0.1), with a restart because a network that overshot may be stuck in saturated or dead regions. (`doctor/rules.py → learning_rate_too_high`)

**12. And too low?**
After ≥150 steps, the last 30 snapshots improved loss by <1% while the loss is still high (>0.45) and accuracy <95%.

**13. What is L2 regularisation and how did you implement it?**
Add `½·λ·Σw²` to the loss, which adds `λw` to each weight's gradient (weight decay); it penalises large weights, smoothing the function. It is a live-adjustable setting.

**14. Batch size trade-offs?**
Small: noisy gradients (regularising), more updates per epoch, poor hardware use. Large: smooth gradients, better hardware use, fewer updates per epoch.

**15. What is an epoch, and how do you compute it?**
One pass over the training set: `step × batch_size / n_train`.

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
tanh/sigmoid outputs pinned at the flat ends have ~0 slope. But a converged spiral network legitimately saturates; the rule now only fires while loss is still high (≥ 0.25). Found by manual testing, locked by a unit test.

**23. How do you detect overfitting from telemetry?**
Test loss minus train loss above 0.15, accuracy gap above 5%, and test loss risen more than 10% from its minimum. Evidence jumps to the step where test loss was lowest.

**24. Why isn't loss alone enough to debug a model?**
It doesn't say where or why training fails. Per-layer gradient, activation and weight statistics localise the problem.

**25. A spiral network sometimes just doesn't learn. Why, and what did you do?**
A 2-layer tanh net on a spiral is init-sensitive: with seed 7 the loss sits at ~0.66 for thousands of steps, with seeds 1, 3, 4 and 10 it learns. I measured it, changed the default seed to one that learns reliably, and documented it rather than hiding it. It is also a nice story for the Compare page (same data, different init).

## CNNs and MNIST

**26. What does a convolution do?**
Slides a small filter over the input computing a dot product at each position, producing a feature map showing where a pattern occurs. Weights are shared across positions.

**27. How did you implement convolution without a framework?**
im2col: gather every receptive field into a row of a big matrix, so the convolution is one matrix multiplication with the filter matrix. The backward pass does the matrix products in reverse and scatters the input gradient back with col2im (summing where windows overlap). (`learnscope/mnist/layers.py`)

**28. Why are CNNs better for images than MLPs?**
Weight sharing (fewer parameters), locality (nearby pixels matter together), and translation equivariance. Here: 5,386 CNN parameters vs 109,386 for the MLP, with the CNN reaching 96.9% against the MLP's ~94% on the same 10k images.

**29. What does max-pooling do, and what is its gradient?**
Takes the max in each window, shrinking the map and giving small-shift tolerance. The gradient flows only to the winning position; when there are ties it is split among them (tested).

**30. How are the CNN's output sizes computed?**
28×28 →(conv 5×5, valid) 24×24×8 →(pool) 12×12×8 →(conv 3×3) 10×10×16 →(pool) 5×5×16 → flatten 400 → dense 10.

**31. How do you read a confusion matrix?**
Rows are the true class, columns the prediction; the diagonal is correct; bright off-diagonal cells show systematic confusions (4↔9, 3↔5).

**32. How does the draw-a-digit pad make drawings look like MNIST?**
Crop to the drawn area, fit it (aspect ratio kept) into a 20×20 box inside a 28×28 frame, then shift so the centre of mass is at the centre, as MNIST was prepared. Without this, predictions are poor. (`learnscope/mnist/draw.py`, tested for centring and for filling the box whatever the drawing size)

**33. Why a subset of MNIST?**
10k train + 2k test is enough to show learning, ships as a 1.9 MB `.npz` file inside the package, trains an epoch in seconds on a CPU, and needs no download at run time.

**34. What happens if you train longer on a small subset?**
Train accuracy approaches 100% while test accuracy plateaus. The MLP shows it clearly: it peaks around 94.4% after ~3 epochs and drifts down to 93.6% by epoch 10.

## Python, concurrency, performance

**35. Why train on a background thread?**
Training would block the event loop that serves the UI and make every control feel frozen. The `Run` owns the model on its own thread; the UI only sends commands and reads snapshots.

**36. How do the UI and the trainer share state safely?**
One `RLock` per run. The UI reads through methods that copy under the lock (`series()`, `view_snapshot()`); it never touches live arrays. It polls `run.version` and redraws only when that changed.

**37. What does the GIL do to this design?**
Tiny networks spend most of their time in Python, not in NumPy kernels that release the GIL, so the trainer and the UI contend for one interpreter lock. I lowered the switch interval (`sys.setswitchinterval(0.001)`) so the trainer gets the lock back quickly, and made the UI work cheap.

**38. What performance bug did you find and how?**
Building validated `plotly.graph_objects.Figure` objects cost 40-60 ms each. With ~9 charts per refresh the UI thread starved the trainer: 25 steps/s. I profiled it, switched to plain dict figures (microseconds) and updated plots in place: ~86 steps/s with the UI attached. A test now guards that building a whole refresh set stays under 25 ms.

**39. Is NumPy fast for the playground's tiny networks?**
Not especially: a 2-8-8-1 network is latency-bound by Python and NumPy call overhead, not by arithmetic (about 0.15 ms per step). NumPy (and BLAS) pays off on MNIST, where the matrices are big. A GPU would not help the toy networks either.

**40. How would you scale it to many users?**
Each open tab runs a training thread, so the first limit is CPU. Options: cap runs per server, share results for identical configs (a config is hashable data), move training to worker processes, or precompute demo runs.

**41. Why NiceGUI and Plotly?**
One language end to end and no JavaScript build step, so the whole project is readable and testable by a Python developer. Plotly gives interactive charts from a JSON description; chart builders are pure functions returning dicts, so they are tested without a browser.

## Design and engineering decisions

**42. How does time travel work?**
Each snapshot stores weights and per-layer statistics. Every view is a pure function of "the selected snapshot", so scrubbing re-renders consistently. The decision boundary is recomputed from stored weights.

**43. How do you bound memory for the history?**
A buffer with item and byte limits that thins the oldest half (keeps every second item) rather than discarding the start of training. A default snapshot is ~3.3 KB.

**44. How do you prevent races between reconfiguring and training?**
`Run._build` first pauses (stops and joins the thread), then swaps in the new trainer and clears the history under the lock, then bumps `version` so the UI redraws; `set_config` for live keys (lr, L2, batch size) mutates the trainer under the same lock without a restart.

**45. How are shareable links implemented and secured?**
The config is JSON → base64url in the query string. On load it is **sanitised**: unknown values fall back to defaults, numbers are clamped, the layer count is capped and the string length is limited, so a malicious link can't crash the app or request absurd sizes. (`learnscope/config.py`, tested)

**46. How do you test ML code deterministically?**
Seeded RNG for data and initialisation; numerical gradient checks; independent reference implementations (a separate forward pass vs the snapshot's numbers); synthetic telemetry histories for the Doctor instead of real training runs.

**47. What would you do next?**
A tiny transformer with attention visualisation, experiment tracking across sessions, and process-based workers so many users do not share one GIL.

**48. What was the hardest bug?**
Pick your own from this list and say what you did: (a) the Doctor debounce that starved at full speed; (b) the false-positive saturation rule on a healthy spiral run; (c) UI figure building starving the training thread (see Q38); (d) the init-sensitive spiral (Q25).

**49. How is accessibility handled?**
Labelled controls and sliders, `aria-label`s on timeline and diagnosis cards, an "ⓘ What am I looking at?" on every chart, keyboard shortcuts, and a light/dark theme toggle.

**50. What are the limits of the Model Doctor?**
It is heuristic: thresholds were tuned on a handful of runs, so it can miss or misjudge unusual setups. Being rule-based makes it explainable and unit-testable, and each rule documents its evidence.

## Landscape, comparison, debugging

**51. What is a loss landscape slice and what are its limits?**
The loss depends on every weight; we pick two random orthogonal directions, scale them per layer to the layer's weight norm (so slices are comparable across networks) and plot the loss on that plane. It is a 2-D shadow of a huge space: valleys can look flatter or sharper than they are, and the training path projected onto it can leave the picture. (`learnscope/landscape.py`)

**52. Why make the two directions orthogonal?**
So that projecting a weight vector onto the plane gives exact coordinates; unit test: a point built as c + 0.3·d1 − 0.2·d2 projects back to (0.3, −0.2).

**53. How does Compare keep the comparison fair?**
Both runs share the dataset, split and seed (and the initialiser seed), and the defaults change exactly one idea (Adam vs SGD). Changing the data on either side changes it on both. Differences therefore come from the setting under test.

**54. Tell me about a bug you found by testing at the real update rate.**
The Doctor used a debounce, which starved when snapshots arrive every ~30 ms; it only worked when I slowed the run down. I changed it to a throttle, added a regression test, and verified that the test fails on the old code. Lesson: a test that cannot fail proves nothing.

**55. You rewrote the project from TypeScript to Python. What did you gain and lose?**
Gained: one language end to end, an engine that is easy to read and test (hand-written NumPy backprop and convolutions instead of library calls), and a simpler stack with no JavaScript build. Lost: in-browser GPU training, a static-site deployment (it now needs a Python server), and the 60 fps redraw (it now updates a few times per second over a websocket). I measured both versions rather than assuming.
