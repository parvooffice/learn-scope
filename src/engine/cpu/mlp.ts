import { gaussian, mulberry32 } from '../../lib/rng'

/**
 * A dependency-free MLP with hand-written backpropagation (tanh hidden layers, sigmoid output,
 * binary cross-entropy, full-batch Adam). It powers the landing-page animation without loading
 * TensorFlow.js, and doubles as an independent check of the gradient maths in tests.
 */
export class CpuMLP {
  readonly sizes: number[]
  W: Float64Array[] = []
  b: Float64Array[] = []
  private mW: Float64Array[] = []
  private vW: Float64Array[] = []
  private mB: Float64Array[] = []
  private vB: Float64Array[] = []
  private t = 0

  constructor(sizes: number[], seed = 1) {
    this.sizes = sizes
    const rng = mulberry32(seed)
    for (let l = 0; l < sizes.length - 1; l++) {
      const nin = sizes[l]
      const nout = sizes[l + 1]
      const std = Math.sqrt(2 / (nin + nout))
      this.W.push(Float64Array.from({ length: nin * nout }, () => gaussian(rng) * std))
      this.b.push(new Float64Array(nout))
      this.mW.push(new Float64Array(nin * nout))
      this.vW.push(new Float64Array(nin * nout))
      this.mB.push(new Float64Array(nout))
      this.vB.push(new Float64Array(nout))
    }
  }

  /** Forward pass for one point; returns P(class 1). */
  predict(x0: number, x1: number): number {
    let a: Float64Array = Float64Array.of(x0, x1)
    const L = this.W.length
    for (let l = 0; l < L; l++) {
      const nin = this.sizes[l]
      const nout = this.sizes[l + 1]
      const out = new Float64Array(nout)
      for (let j = 0; j < nout; j++) {
        let z = this.b[l][j]
        for (let i = 0; i < nin; i++) z += a[i] * this.W[l][i * nout + j]
        out[j] = l === L - 1 ? 1 / (1 + Math.exp(-z)) : Math.tanh(z)
      }
      a = out
    }
    return a[0]
  }

  /** Mean binary cross-entropy and its gradients over the whole dataset. */
  gradients(x: ArrayLike<number>, y: ArrayLike<number>) {
    const n = y.length
    const L = this.W.length
    // forward, keeping every layer's activations
    const acts: Float64Array[] = [Float64Array.from(x as ArrayLike<number>)]
    for (let l = 0; l < L; l++) {
      const nin = this.sizes[l]
      const nout = this.sizes[l + 1]
      const prev = acts[l]
      const out = new Float64Array(n * nout)
      for (let s = 0; s < n; s++) {
        for (let j = 0; j < nout; j++) {
          let z = this.b[l][j]
          for (let i = 0; i < nin; i++) z += prev[s * nin + i] * this.W[l][i * nout + j]
          out[s * nout + j] = l === L - 1 ? 1 / (1 + Math.exp(-z)) : Math.tanh(z)
        }
      }
      acts.push(out)
    }
    let loss = 0
    const last = acts[L]
    for (let s = 0; s < n; s++) {
      const p = Math.min(1 - 1e-12, Math.max(1e-12, last[s]))
      loss += -(y[s] * Math.log(p) + (1 - y[s]) * Math.log(1 - p))
    }
    loss /= n

    // backward: for sigmoid+BCE, dL/dz at the output is simply (p - y) / n
    const dW = this.W.map((w) => new Float64Array(w.length))
    const dB = this.b.map((v) => new Float64Array(v.length))
    let dz = new Float64Array(n)
    for (let s = 0; s < n; s++) dz[s] = (last[s] - y[s]) / n
    for (let l = L - 1; l >= 0; l--) {
      const nin = this.sizes[l]
      const nout = this.sizes[l + 1]
      const prev = acts[l]
      for (let s = 0; s < n; s++) {
        for (let j = 0; j < nout; j++) {
          const d = dz[s * nout + j]
          dB[l][j] += d
          for (let i = 0; i < nin; i++) dW[l][i * nout + j] += prev[s * nin + i] * d
        }
      }
      if (l > 0) {
        const nd = new Float64Array(n * nin)
        for (let s = 0; s < n; s++) {
          for (let i = 0; i < nin; i++) {
            let g = 0
            for (let j = 0; j < nout; j++) g += dz[s * nout + j] * this.W[l][i * nout + j]
            const a = prev[s * nin + i]
            nd[s * nin + i] = g * (1 - a * a) // tanh'
          }
        }
        dz = nd
      }
    }
    return { loss, dW, dB }
  }

  /** One full-batch Adam step. Returns the loss before the update. */
  trainStep(x: ArrayLike<number>, y: ArrayLike<number>, lr: number): number {
    const { loss, dW, dB } = this.gradients(x, y)
    this.t++
    const b1 = 0.9
    const b2 = 0.999
    const c1 = 1 - Math.pow(b1, this.t)
    const c2 = 1 - Math.pow(b2, this.t)
    const upd = (p: Float64Array, g: Float64Array, m: Float64Array, v: Float64Array) => {
      for (let i = 0; i < p.length; i++) {
        m[i] = b1 * m[i] + (1 - b1) * g[i]
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]
        p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8)
      }
    }
    for (let l = 0; l < this.W.length; l++) {
      upd(this.W[l], dW[l], this.mW[l], this.vW[l])
      upd(this.b[l], dB[l], this.mB[l], this.vB[l])
    }
    return loss
  }
}
