import { evaluate, type LayerWeights } from './forward'
import type { LayerSpec } from './types'
import { gaussian, mulberry32 } from '../lib/rng'

/**
 * Loss landscape slice (after Li et al., "Visualizing the Loss Landscape of Neural Nets").
 *
 * The loss is a function of ALL weights, which is far too many dimensions to draw. So we pick two
 * random directions d1, d2 in weight space and plot the loss on the 2-D plane
 *      w(α, β) = w_center + α·d1 + β·d2
 * The directions are scaled layer by layer to the size of that layer's weights, so "α = 1" means
 * "change every layer by about 100% of its own size", which makes slices comparable across nets.
 */

export function flatten(w: LayerWeights[]): Float64Array {
  let n = 0
  for (const l of w) n += l.W.length + l.b.length
  const out = new Float64Array(n)
  let o = 0
  for (const l of w) {
    for (let i = 0; i < l.W.length; i++) out[o++] = l.W[i]
    for (let i = 0; i < l.b.length; i++) out[o++] = l.b[i]
  }
  return out
}

export function unflatten(template: LayerWeights[], flat: ArrayLike<number>): LayerWeights[] {
  let o = 0
  return template.map((l) => {
    const W = new Float32Array(l.W.length)
    const b = new Float32Array(l.b.length)
    for (let i = 0; i < W.length; i++) W[i] = flat[o++]
    for (let i = 0; i < b.length; i++) b[i] = flat[o++]
    return { W, b }
  })
}

/** Start/end offsets of each layer's parameters inside the flat vector. */
function layerRanges(template: LayerWeights[]): [number, number][] {
  const out: [number, number][] = []
  let o = 0
  for (const l of template) {
    const n = l.W.length + l.b.length
    out.push([o, o + n])
    o += n
  }
  return out
}

const dot = (a: Float64Array, b: Float64Array) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

/** Two orthogonal random directions, each rescaled per layer to that layer's weight norm. */
export function randomDirections(
  center: LayerWeights[],
  seed: number,
): { d1: Float64Array; d2: Float64Array } {
  const rng = mulberry32(seed * 6151 + 17)
  const c = flatten(center)
  const ranges = layerRanges(center)
  const make = () => Float64Array.from({ length: c.length }, () => gaussian(rng))
  const normalise = (d: Float64Array) => {
    for (const [s, e] of ranges) {
      let dn = 0
      let cn = 0
      for (let i = s; i < e; i++) {
        dn += d[i] * d[i]
        cn += c[i] * c[i]
      }
      const k = dn > 0 ? Math.sqrt(cn || e - s) / Math.sqrt(dn) : 0
      for (let i = s; i < e; i++) d[i] *= k
    }
  }
  const d1 = make()
  const d2 = make()
  // Gram–Schmidt so that projections onto (d1, d2) are exact coordinates
  const k = dot(d2, d1) / dot(d1, d1)
  for (let i = 0; i < d2.length; i++) d2[i] -= k * d1[i]
  normalise(d1)
  normalise(d2)
  // per-layer rescaling slightly breaks orthogonality; re-orthogonalise once more
  const k2 = dot(d2, d1) / dot(d1, d1)
  for (let i = 0; i < d2.length; i++) d2[i] -= k2 * d1[i]
  return { d1, d2 }
}

/** Coordinates (α, β) of a weight vector in the slice plane. */
export function project(
  w: Float64Array,
  center: Float64Array,
  d1: Float64Array,
  d2: Float64Array,
): { a: number; b: number } {
  let a = 0
  let b = 0
  let n1 = 0
  let n2 = 0
  for (let i = 0; i < w.length; i++) {
    const d = w[i] - center[i]
    a += d * d1[i]
    b += d * d2[i]
    n1 += d1[i] * d1[i]
    n2 += d2[i] * d2[i]
  }
  return { a: a / n1, b: b / n2 }
}

export interface Landscape {
  G: number
  range: number
  /** loss at each grid cell; row 0 is β = +range (top of the picture) */
  values: Float32Array
  min: number
  /** loss value used as the top of the colour scale (95th percentile, so one spike can't wash it out) */
  clip: number
  centerLoss: number
  d1: Float64Array
  d2: Float64Array
}

export interface LandscapeOptions {
  G?: number
  range?: number
  seed?: number
  onProgress?: (fraction: number) => void
  signal?: { cancelled: boolean }
}

const tick = () => new Promise<void>((r) => setTimeout(r, 0))

/** Evaluate the loss on a G×G grid around `center` (G should be odd so the centre cell is exact). */
export async function computeLandscape(
  specs: LayerSpec[],
  center: LayerWeights[],
  x: Float32Array,
  y: Uint8Array,
  opts: LandscapeOptions = {},
): Promise<Landscape | null> {
  const G = opts.G ?? 31
  const range = opts.range ?? 1
  const { d1, d2 } = randomDirections(center, opts.seed ?? 1)
  const c = flatten(center)
  const values = new Float32Array(G * G)
  const w = new Float64Array(c.length)
  let lastYield = performance.now()

  for (let gy = 0; gy < G; gy++) {
    const beta = range - (gy / (G - 1)) * 2 * range
    for (let gx = 0; gx < G; gx++) {
      const alpha = -range + (gx / (G - 1)) * 2 * range
      for (let i = 0; i < w.length; i++) w[i] = c[i] + alpha * d1[i] + beta * d2[i]
      values[gy * G + gx] = evaluate(specs, unflatten(center, w), x, y).loss
    }
    opts.onProgress?.((gy + 1) / G)
    // keep the page responsive, but only pause when ~30 ms of work has piled up
    if (performance.now() - lastYield > 30) {
      await tick()
      lastYield = performance.now()
    }
    if (opts.signal?.cancelled) return null
  }

  const sorted = Array.from(values).sort((a, b) => a - b)
  return {
    G,
    range,
    values,
    min: sorted[0],
    clip: sorted[Math.floor(sorted.length * 0.95)],
    centerLoss: evaluate(specs, center, x, y).loss,
    d1,
    d2,
  }
}
