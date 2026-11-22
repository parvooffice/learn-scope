import { Value } from './value'
import { mulberry32 } from '../../lib/rng'

/**
 * A tiny 2-3-1 network used by the Backprop explainer:
 *   z_j = w_j1·x1 + w_j2·x2 + b_j      (3 hidden neurons)
 *   h_j = tanh(z_j)
 *   ŷ   = v_1·h_1 + v_2·h_2 + v_3·h_3 + c
 *   L   = (ŷ - t)²
 * Every number the explainer shows is computed here, live, by the autograd engine.
 */
export interface ExplainerParams {
  x: [number, number]
  t: number
  /** w[j][i]: weight from input i to hidden neuron j */
  w: number[][]
  b: number[]
  v: number[]
  c: number
}

export interface Grads {
  yhat: number
  v: number[]
  c: number
  h: number[]
  z: number[]
  w: number[][]
  b: number[]
}

export interface ExplainerResult {
  z: number[]
  h: number[]
  yhat: number
  loss: number
  /** gradients found by the autograd engine (Value.backward) */
  auto: Grads
  /** the same gradients written out by hand with the chain rule */
  manual: Grads
  /** largest |auto - manual| over all gradients; should be ~1e-16 */
  maxDiff: number
}

export function initialParams(seed = 4): ExplainerParams {
  const rng = mulberry32(seed)
  const r = () => Number((rng() * 2 - 1).toFixed(2))
  return {
    x: [0.5, -0.8],
    t: 0.7,
    w: [
      [r(), r()],
      [r(), r()],
      [r(), r()],
    ],
    b: [0.1, -0.1, 0.05],
    v: [r(), r(), r()],
    c: 0,
  }
}

export function runExplainer(p: ExplainerParams): ExplainerResult {
  // build the computation graph with the autograd engine
  const x = p.x.map((e, i) => new Value(e, [], '', `x${i + 1}`))
  const w = p.w.map((row, j) => row.map((e, i) => new Value(e, [], '', `w${j + 1}${i + 1}`)))
  const b = p.b.map((e, j) => new Value(e, [], '', `b${j + 1}`))
  const v = p.v.map((e, j) => new Value(e, [], '', `v${j + 1}`))
  const c = new Value(p.c, [], '', 'c')

  const z = w.map((row, j) => row[0].mul(x[0]).add(row[1].mul(x[1])).add(b[j]))
  const h = z.map((zj) => zj.tanh())
  const yhat = v[0].mul(h[0]).add(v[1].mul(h[1])).add(v[2].mul(h[2])).add(c)
  const loss = yhat.sub(p.t).pow(2)
  loss.backward()

  const auto: Grads = {
    yhat: yhat.grad,
    v: v.map((e) => e.grad),
    c: c.grad,
    h: h.map((e) => e.grad),
    z: z.map((e) => e.grad),
    w: w.map((row) => row.map((e) => e.grad)),
    b: b.map((e) => e.grad),
  }

  // the chain rule, by hand
  const hv = h.map((e) => e.data)
  const dyhat = 2 * (yhat.data - p.t)
  const dh = p.v.map((vj) => dyhat * vj)
  const dz = dh.map((d, j) => d * (1 - hv[j] * hv[j]))
  const manual: Grads = {
    yhat: dyhat,
    v: hv.map((hj) => dyhat * hj),
    c: dyhat,
    h: dh,
    z: dz,
    w: dz.map((d) => [d * p.x[0], d * p.x[1]]),
    b: dz.slice(),
  }

  const flat = (g: Grads) => [g.yhat, g.c, ...g.v, ...g.h, ...g.z, ...g.b, ...g.w.flat()]
  const fa = flat(auto)
  const fm = flat(manual)
  const maxDiff = Math.max(...fa.map((a, i) => Math.abs(a - fm[i])))

  return { z: z.map((e) => e.data), h: hv, yhat: yhat.data, loss: loss.data, auto, manual, maxDiff }
}

/** One gradient-descent step: param ← param − lr · gradient. */
export function applyStep(p: ExplainerParams, g: Grads, lr: number): ExplainerParams {
  return {
    ...p,
    w: p.w.map((row, j) => row.map((e, i) => e - lr * g.w[j][i])),
    b: p.b.map((e, j) => e - lr * g.b[j]),
    v: p.v.map((e, j) => e - lr * g.v[j]),
    c: p.c - lr * g.c,
  }
}
