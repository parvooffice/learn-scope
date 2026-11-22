import type { Activation, LayerSpec } from './types'

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI)

/** Activation functions, written out by hand so the worker (TF.js) and the UI agree exactly. */
export function actFn(a: Activation, z: number): number {
  switch (a) {
    case 'relu':
      return z > 0 ? z : 0
    case 'tanh':
      return Math.tanh(z)
    case 'sigmoid':
      return 1 / (1 + Math.exp(-z))
    case 'gelu':
      // tanh approximation of GELU
      return 0.5 * z * (1 + Math.tanh(SQRT_2_OVER_PI * (z + 0.044715 * z * z * z)))
  }
}

export interface LayerWeights {
  /** row-major [nin][nout] */
  W: Float32Array
  b: Float32Array
}

/** Forward one point through every layer; returns each layer's activations (output last). */
export function forwardPoint(
  specs: LayerSpec[],
  weights: LayerWeights[],
  x0: number,
  x1: number,
): Float32Array[] {
  const out: Float32Array[] = []
  let prev: ArrayLike<number> = [x0, x1]
  for (let l = 0; l < specs.length; l++) {
    const { nin, nout, activation } = specs[l]
    const { W, b } = weights[l]
    const a = new Float32Array(nout)
    for (let j = 0; j < nout; j++) {
      let z = b[j]
      for (let i = 0; i < nin; i++) z += prev[i] * W[i * nout + j]
      a[j] = actFn(activation, z)
    }
    out.push(a)
    prev = a
  }
  return out
}

/**
 * Evaluate every unit of every layer on a G×G grid covering [-extent, extent]².
 * Result[l] is a Float32Array of length G*G*nout (unit-major per cell).
 * Used for the decision boundary (last layer) and the per-neuron previews.
 */
export function forwardGrid(
  specs: LayerSpec[],
  weights: LayerWeights[],
  G: number,
  extent: number,
): Float32Array[] {
  const layers = specs.map((s) => new Float32Array(G * G * s.nout))
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const x0 = -extent + ((gx + 0.5) / G) * 2 * extent
      // row 0 is the top of the picture => largest y
      const x1 = extent - ((gy + 0.5) / G) * 2 * extent
      const acts = forwardPoint(specs, weights, x0, x1)
      const cell = gy * G + gx
      for (let l = 0; l < specs.length; l++) {
        layers[l].set(acts[l], cell * specs[l].nout)
      }
    }
  }
  return layers
}

/** Accuracy and BCE loss of the network on a dataset (CPU reference implementation). */
export function evaluate(
  specs: LayerSpec[],
  weights: LayerWeights[],
  x: Float32Array,
  y: Uint8Array,
): { loss: number; acc: number } {
  let loss = 0
  let correct = 0
  const n = y.length
  for (let i = 0; i < n; i++) {
    const acts = forwardPoint(specs, weights, x[i * 2], x[i * 2 + 1])
    const p = Math.min(1 - 1e-7, Math.max(1e-7, acts[acts.length - 1][0]))
    loss += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p))
    if ((p > 0.5 ? 1 : 0) === y[i]) correct++
  }
  return { loss: loss / n, acc: correct / n }
}
