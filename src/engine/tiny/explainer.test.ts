import { describe, expect, it } from 'vitest'
import { applyStep, initialParams, runExplainer } from './explainer'

describe('backprop explainer engine', () => {
  it('hand-written chain rule equals autograd for every gradient', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const r = runExplainer(initialParams(seed))
      expect(r.maxDiff).toBeLessThan(1e-12)
    }
  })

  it('forward values follow the formulas', () => {
    const p = initialParams(4)
    const r = runExplainer(p)
    const z0 = p.w[0][0] * p.x[0] + p.w[0][1] * p.x[1] + p.b[0]
    expect(r.z[0]).toBeCloseTo(z0, 12)
    expect(r.h[0]).toBeCloseTo(Math.tanh(z0), 12)
    const yhat = p.v.reduce((s, vj, j) => s + vj * r.h[j], 0) + p.c
    expect(r.yhat).toBeCloseTo(yhat, 12)
    expect(r.loss).toBeCloseTo((yhat - p.t) ** 2, 12)
  })

  it('gradients match numerical differentiation', () => {
    const p = initialParams(4)
    const r = runExplainer(p)
    const h = 1e-6
    const bump = (fn: (q: typeof p, d: number) => void) => {
      const a = structuredClone(p)
      const b = structuredClone(p)
      fn(a, h)
      fn(b, -h)
      return (runExplainer(a).loss - runExplainer(b).loss) / (2 * h)
    }
    expect(r.auto.w[1][0]).toBeCloseTo(
      bump((q, d) => (q.w[1][0] += d)),
      6,
    )
    expect(r.auto.v[2]).toBeCloseTo(
      bump((q, d) => (q.v[2] += d)),
      6,
    )
    expect(r.auto.b[0]).toBeCloseTo(
      bump((q, d) => (q.b[0] += d)),
      6,
    )
    expect(r.auto.c).toBeCloseTo(
      bump((q, d) => (q.c += d)),
      6,
    )
  })

  it('a small gradient step lowers the loss', () => {
    let p = initialParams(4)
    const before = runExplainer(p)
    p = applyStep(p, before.auto, 0.05)
    expect(runExplainer(p).loss).toBeLessThan(before.loss)
  })

  it('repeated steps drive the loss toward zero', () => {
    let p = initialParams(4)
    for (let i = 0; i < 300; i++) p = applyStep(p, runExplainer(p).auto, 0.1)
    expect(runExplainer(p).loss).toBeLessThan(1e-4)
  })
})
