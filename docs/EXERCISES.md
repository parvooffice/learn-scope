# Your turn: exercises

These are the parts worth writing **yourself**, because they are exactly what an interviewer will ask you to derive or sketch.
The finished versions already exist in the repo; **delete the body, re-write it, and use the provided test to check yourself.**
Commit your own version only after the tests pass.

## 1. Autograd: backward of `mul` and `tanh`

File: `src/engine/tiny/value.ts`

```ts
mul(other: number | Value): Value {
  const o = Value.of(other)
  const out = new Value(this.data * o.data, [this, o], '×')
  out._backward = () => {
    // TODO: how much does the output change when `this` changes? when `o` changes?
    // remember: multiply by out.grad (chain rule) and ACCUMULATE with +=
  }
  return out
}
```

Hints: for `out = a·b`, `∂out/∂a = b` and `∂out/∂b = a`. For `out = tanh(a)`, `∂out/∂a = 1 − out²`.

Check: `npx vitest run src/engine/tiny` — the numerical-gradient tests (`tanh`, `add / mul`, whole MLP) must stay green.
Bonus: break `+=` into `=` and read which test fails and why.

## 2. A dense layer's forward pass (no libraries)

File: `src/engine/forward.ts` → `forwardPoint`

Write, for one layer, `a[j] = activation( b[j] + Σ_i x[i] · W[i][j] )` where `W` is stored row-major as `W[i * nout + j]`.

Check: `npx vitest run src/lib/lib.test.ts` ("CPU forward pass matches a hand calculation") and
`src/engine/tf/trainer.test.ts` ("TF metrics agree with the independent CPU forward pass").

Bonus: compute by hand what `forwardPoint(specs, weights, 1, 0.5)` returns in the test and confirm.

## 3. A Model Doctor rule, end to end

Write a new rule **"loss is NaN-free but accuracy is stuck at 50%"** (the net is not learning at all):

1. Create `src/doctor/rules/stuck.ts` exporting `const stuck: Rule`.
2. Fire when there are ≥ 40 snapshots, the last 20 all have `trainAcc` within ±0.03 of 0.5, and `trainLoss` is within 0.02 of `ln 2 ≈ 0.693`.
3. Evidence: the step where the stretch begins; chart `'loss'`. Fix: raise the learning rate ×3 (`reset: false`), or if `lr ≥ 0.3`, switch the optimizer to Adam (`reset: true`).
4. Register it in `src/doctor/doctor.ts`.
5. Write tests in `src/doctor/doctor.test.ts` using `mkHistory`: one history that must trigger, one healthy history that must not, one too-short history that must not.

Check: `npx vitest run src/doctor`.

## 4. Read-and-explain (no code)

Open `src/engine/tf/trainer.ts` and, in your own words, write down:

- why `snapshot()` computes the gradient on the **full** training set, and what it costs;
- why `trainStep` wraps work in `tf.tidy`, and what would happen without it;
- what Adam's `m` and `v` are and why there is a bias-correction term (`1 − β^t`).
