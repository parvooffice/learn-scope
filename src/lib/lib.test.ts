import { describe, expect, it } from 'vitest'
import { HistoryBuffer } from './ringBuffer'
import { decodeConfig, encodeConfig, sanitizeConfig } from './urlConfig'
import { histogram, meanStd } from './stats'
import { DEFAULT_CONFIG } from '../engine/types'
import { generateDataset, splitDataset } from '../engine/datasets/toy'
import { evaluate, forwardGrid, forwardPoint } from '../engine/forward'
import { layerSpecs } from '../engine/types'

describe('dataset generators', () => {
  const kinds = ['moons', 'circles', 'spiral', 'xor', 'blobs'] as const
  for (const k of kinds) {
    it(`${k}: right size, both classes, bounded, deterministic`, () => {
      const a = generateDataset(k, 300, 0.1, 5)
      const b = generateDataset(k, 300, 0.1, 5)
      expect(a.n).toBe(300)
      expect(a.x.length).toBe(600)
      expect(Array.from(a.x)).toEqual(Array.from(b.x))
      const ones = a.y.reduce((s, v) => s + v, 0)
      expect(ones).toBeGreaterThan(90)
      expect(ones).toBeLessThan(210)
      expect(Math.max(...Array.from(a.x).map(Math.abs))).toBeLessThan(2.5)
    })
  }
  it('different seeds give different data', () => {
    const a = generateDataset('moons', 50, 0.1, 1)
    const b = generateDataset('moons', 50, 0.1, 2)
    expect(Array.from(a.x)).not.toEqual(Array.from(b.x))
  })
  it('split keeps every point exactly once', () => {
    const ds = generateDataset('circles', 200, 0.05, 3)
    const { train, test } = splitDataset(ds, 0.25, 3)
    expect(test.n).toBe(50)
    expect(train.n).toBe(150)
    expect(train.n + test.n).toBe(ds.n)
  })
})

describe('HistoryBuffer', () => {
  it('thins old items but keeps first and last when the item limit is hit', () => {
    const buf = new HistoryBuffer<number>(() => 1, 10, 1e9)
    for (let i = 0; i < 100; i++) buf.push(i)
    expect(buf.length).toBeLessThanOrEqual(10)
    expect(buf.at(0)).toBe(0)
    expect(buf.last()).toBe(99)
    const arr = buf.toArray()
    for (let i = 1; i < arr.length; i++) expect(arr[i]).toBeGreaterThan(arr[i - 1])
  })
  it('respects the byte limit', () => {
    const buf = new HistoryBuffer<number>(() => 100, 1e9, 1000)
    for (let i = 0; i < 200; i++) buf.push(i)
    expect(buf.totalBytes).toBeLessThanOrEqual(1000)
    expect(buf.last()).toBe(199)
  })
  it('nearest finds the closest key', () => {
    const buf = new HistoryBuffer<number>(() => 1)
    ;[0, 10, 20, 30].forEach((n) => buf.push(n))
    expect(buf.nearest(14, (x) => x)).toBe(1)
    expect(buf.nearest(16, (x) => x)).toBe(2)
    expect(buf.nearest(1000, (x) => x)).toBe(3)
  })
})

describe('URL config encoding', () => {
  it('round-trips', () => {
    const cfg = { ...DEFAULT_CONFIG, lr: 0.5, dataset: 'xor' as const }
    expect(decodeConfig(encodeConfig(cfg))).toEqual(cfg)
  })
  it('returns null for garbage and repairs bad values', () => {
    expect(decodeConfig('%%%not-base64')).toBeNull()
    const fixed = sanitizeConfig({
      lr: 1e9,
      layers: [{ units: 999, activation: 'evil' }],
      dataset: 'nope',
    })
    expect(fixed.lr).toBeLessThanOrEqual(3)
    expect(fixed.layers[0].units).toBe(16)
    expect(fixed.layers[0].activation).toBe('tanh')
    expect(fixed.dataset).toBe(DEFAULT_CONFIG.dataset)
  })
})

describe('stats', () => {
  it('meanStd', () => {
    const r = meanStd([1, 2, 3, 4])
    expect(r.mean).toBeCloseTo(2.5)
    expect(r.std).toBeCloseTo(Math.sqrt(1.25))
  })
  it('histogram counts every value', () => {
    const h = histogram([0, 0.1, 0.2, 0.9, 1], 4)
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(5)
    expect(h.lo).toBe(0)
    expect(h.hi).toBe(1)
  })
  it('histogram of constant data does not crash', () => {
    const h = histogram([3, 3, 3], 5)
    expect(h.counts.reduce((a, b) => a + b, 0)).toBe(3)
  })
})

describe('CPU forward pass', () => {
  const specs = layerSpecs({ layers: [{ units: 2, activation: 'relu' }] })
  // hand-computed network: h = relu([x0 - x1, x0 + x1]); out = sigmoid(h0 - h1 + 0.5)
  const weights = [
    { W: new Float32Array([1, 1, -1, 1]), b: new Float32Array([0, 0]) },
    { W: new Float32Array([1, -1]), b: new Float32Array([0.5]) },
  ]
  it('matches a hand calculation', () => {
    const acts = forwardPoint(specs, weights, 1, 0.5)
    expect(acts[0][0]).toBeCloseTo(0.5) // 1 - 0.5
    expect(acts[0][1]).toBeCloseTo(1.5) // 1 + 0.5
    expect(acts[1][0]).toBeCloseTo(1 / (1 + Math.exp(-(0.5 - 1.5 + 0.5))))
  })
  it('grid has the right shape and agrees with pointwise forward', () => {
    const G = 6
    const grid = forwardGrid(specs, weights, G, 1)
    expect(grid[1].length).toBe(G * G)
    // cell (gx=0, gy=0) is top-left => x0 = -1+1/6, x1 = 1-1/6
    const x0 = -1 + 1 / G
    const x1 = 1 - 1 / G
    expect(grid[1][0]).toBeCloseTo(forwardPoint(specs, weights, x0, x1)[1][0], 5)
  })
  it('evaluate returns a sane accuracy', () => {
    const x = new Float32Array([1, 0.5, -1, -0.5])
    const y = new Uint8Array([0, 1])
    const r = evaluate(specs, weights, x, y)
    expect(r.acc).toBeGreaterThanOrEqual(0)
    expect(r.loss).toBeGreaterThan(0)
  })
})
