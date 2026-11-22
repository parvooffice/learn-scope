import { describe, expect, it } from 'vitest'
import { computeLandscape, flatten, project, randomDirections, unflatten } from './landscape'
import { generateDataset } from './datasets/toy'
import { layerSpecs } from './types'
import { evaluate, type LayerWeights } from './forward'
import { mulberry32 } from '../lib/rng'

const specs = layerSpecs({ layers: [{ units: 4, activation: 'tanh' }] })
const rng = mulberry32(3)
const center: LayerWeights[] = specs.map((s) => ({
  W: Float32Array.from({ length: s.nin * s.nout }, () => rng() * 2 - 1),
  b: Float32Array.from({ length: s.nout }, () => rng() * 0.2),
}))
const ds = generateDataset('moons', 60, 0.1, 1)

describe('loss landscape', () => {
  it('flatten/unflatten round-trips', () => {
    const back = unflatten(center, flatten(center))
    back.forEach((l, i) => {
      expect(Array.from(l.W)).toEqual(Array.from(center[i].W))
      expect(Array.from(l.b)).toEqual(Array.from(center[i].b))
    })
  })

  it('directions are orthogonal and layer-scaled', () => {
    const { d1, d2 } = randomDirections(center, 5)
    let dot = 0
    let n1 = 0
    let n2 = 0
    for (let i = 0; i < d1.length; i++) {
      dot += d1[i] * d2[i]
      n1 += d1[i] ** 2
      n2 += d2[i] ** 2
    }
    expect(Math.abs(dot) / Math.sqrt(n1 * n2)).toBeLessThan(1e-6)
    // first layer block of d1 has about the same norm as the first layer of the weights
    const wn = Math.hypot(...Array.from(center[0].W), ...Array.from(center[0].b))
    const dn = Math.hypot(...Array.from(d1.slice(0, center[0].W.length + center[0].b.length)))
    expect(dn / wn).toBeCloseTo(1, 1)
  })

  it('projection recovers the (alpha, beta) used to build a point', () => {
    const { d1, d2 } = randomDirections(center, 9)
    const c = flatten(center)
    const w = c.map((v, i) => v + 0.3 * d1[i] - 0.2 * d2[i])
    const p = project(w, c, d1, d2)
    expect(p.a).toBeCloseTo(0.3, 6)
    expect(p.b).toBeCloseTo(-0.2, 6)
  })

  it('the centre cell equals the loss at the centre weights', async () => {
    const L = await computeLandscape(specs, center, ds.x, ds.y, { G: 9, range: 1 })
    expect(L).not.toBeNull()
    const mid = (9 * 9 - 1) / 2
    const direct = evaluate(specs, center, ds.x, ds.y).loss
    expect(L!.values[mid]).toBeCloseTo(direct, 5)
    expect(L!.centerLoss).toBeCloseTo(direct, 8)
    expect(L!.values.length).toBe(81)
    expect(L!.clip).toBeGreaterThanOrEqual(L!.min)
  })

  it('can be cancelled', async () => {
    const signal = { cancelled: true }
    expect(await computeLandscape(specs, center, ds.x, ds.y, { G: 9, signal })).toBeNull()
  })
})
