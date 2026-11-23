# Your turn: exercises

These are the parts worth writing **yourself**, because they are exactly what an interviewer will ask you to derive or sketch.
The finished versions already exist in the repo; **delete the body, re-write it, and use the tests to check yourself.**
Commit your own version only after the tests pass.

## 1. Autograd: backward of `mul` and `tanh`

File: `learnscope/autograd.py`

```python
def mul(self, other):
    o = Value.of(other)
    out = Value(self.data * o.data, (self, o), "*")

    def _backward():
        # TODO: how much does the output change when `self` changes? when `o` changes?
        # remember: multiply by out.grad (chain rule) and ACCUMULATE with +=
        ...

    out._backward = _backward
    return out
```

Hints: for `out = a * b`, `d(out)/da = b` and `d(out)/db = a`. For `out = tanh(a)`, `d(out)/da = 1 - out**2`.

Check: `python -m pytest tests/test_autograd.py`. The numerical-gradient tests (`tanh`, add / mul, the whole tiny MLP) must stay green.
Bonus: change `+=` into `=` and read which test fails and why (a value used twice gets only one path's gradient).

## 2. A dense layer's forward pass (no libraries beyond NumPy)

File: `learnscope/forward.py` → `forward_all`

Write, for one layer, `a = activation(a_prev @ W + b)` with `W` of shape `(nin, nout)`, the output layer being a sigmoid.
Then do it once with plain Python loops for a single point and compare: `a[j] = act(b[j] + sum_i x[i] * W[i][j])`.

Check: `python -m pytest tests/test_engine.py` ("snapshot contents and agreement with the independent forward pass").
Bonus: compute by hand what a 2-2-1 network with all-zero weights returns for any input, and confirm.

## 3. A Model Doctor rule, end to end

Write a new rule **"loss is NaN-free but accuracy is stuck at 50%"** (the net is not learning at all):

1. Add `stuck(ctx)` to `learnscope/doctor/rules.py`, returning a `Diagnosis` or `None`.
2. Fire when there are at least 40 snapshots, the last 20 all have `train_acc` within ±0.03 of 0.5, and `train_loss` is within 0.02 of `ln 2 ≈ 0.693`.
3. Evidence: the step where the stretch begins, chart `"loss"`. Fix: raise the learning rate ×3 (`reset=False`), or if `lr >= 0.3`, switch the optimizer to Adam (`reset=True`).
4. Register it in the `RULES` list in `learnscope/doctor/__init__.py`.
5. Write tests in `tests/test_doctor.py` using `mk_history` from `tests/helpers.py`: one history that must trigger, one healthy history that must not, one too-short history that must not.

Check: `python -m pytest tests/test_doctor.py`. Watch out: your rule must stay quiet on the existing healthy-run test.

## 4. A new optimizer

File: `learnscope/mlp.py` → `Optimizer`

Add **RMSProp** (`v = 0.9*v + 0.1*g**2; w -= lr * g / (sqrt(v) + 1e-8)`) as a fourth optimizer: add it to `OPTIMIZERS` in
`learnscope/config.py`, give it a label, teach `Optimizer` in the same file about it (it has a hard-coded list of kinds), and extend `test_every_optimizer_reduces_the_loss` in `tests/test_engine.py`.
Question to answer in a sentence: what does Adam add on top of RMSProp?

## 5. Read-and-explain (no code)

Open `learnscope/trainer.py` and, in your own words, write down:

- why `snapshot()` computes the gradient on the **full** training set, and what it costs;
- why the trainer copies the weights into the snapshot instead of referencing them (see the "copies not views" test);
- what Adam's `m` and `v` are and why there is a bias-correction term (`1 - beta**t`).

Then open `learnscope/mnist/layers.py` and explain how `Conv2D.forward` turns a convolution into one matrix multiplication (im2col), and what `backward` has to do to undo it (col2im).
