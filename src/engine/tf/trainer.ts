import type * as TFNS from '@tensorflow/tfjs'
import type { Activation, Dataset, PlaygroundConfig, Snapshot, LayerSpec } from '../types'
import { layerSpecs } from '../types'
import { gaussian, mulberry32 } from '../../lib/rng'
import { buildLayerStats } from './layerStats'

type TF = typeof TFNS

interface OptState {
  m?: TFNS.Tensor
  v?: TFNS.Tensor
}

/**
 * A small MLP trained with a hand-written training loop on top of TensorFlow.js.
 *
 * We deliberately do NOT use model.fit(): we want to see the gradients, control the optimizer,
 * and record statistics at every snapshot. Everything runs on whatever backend TF.js selected
 * (WebGPU, WebGL or CPU).
 */
export class Trainer {
  readonly specs: LayerSpec[]
  step = 0
  lr: number
  l2: number
  batchSize: number

  private readonly tf: TF
  private readonly cfg: PlaygroundConfig
  private readonly W: TFNS.Variable[] = []
  private readonly b: TFNS.Variable[] = []
  private readonly varList: TFNS.Variable[] = []
  private readonly state = new Map<string, OptState>()
  private readonly trainX: TFNS.Tensor2D
  private readonly trainY: TFNS.Tensor2D
  private readonly testX: TFNS.Tensor2D
  private readonly testY: TFNS.Tensor2D
  private readonly nTrain: number
  private perm: Int32Array
  private permPos = 0
  private readonly rng: () => number
  private readonly t0 = Date.now()
  private disposed = false

  constructor(tf: TF, cfg: PlaygroundConfig, train: Dataset, test: Dataset) {
    this.tf = tf
    this.cfg = cfg
    this.lr = cfg.lr
    this.l2 = cfg.l2
    this.batchSize = Math.min(cfg.batchSize, train.n)
    this.specs = layerSpecs(cfg)
    this.rng = mulberry32(cfg.seed * 31 + 1)
    this.nTrain = train.n

    // Initialise weights on the CPU (deterministic across backends), then move to the device.
    const initRng = mulberry32(cfg.seed * 977 + 3)
    const runId = Math.random().toString(36).slice(2, 8)
    this.specs.forEach((s, l) => {
      const std =
        s.activation === 'relu' || s.activation === 'gelu'
          ? Math.sqrt(2 / s.nin) // He init
          : Math.sqrt(2 / (s.nin + s.nout)) // Glorot / Xavier init
      const w = new Float32Array(s.nin * s.nout)
      for (let i = 0; i < w.length; i++) w[i] = gaussian(initRng) * std
      const W = tf.variable(tf.tensor2d(w, [s.nin, s.nout]), true, `${runId}_W${l}`)
      const b = tf.variable(tf.zeros([s.nout]), true, `${runId}_b${l}`)
      this.W.push(W)
      this.b.push(b)
      this.varList.push(W, b)
    })

    this.trainX = tf.tensor2d(train.x, [train.n, 2])
    this.trainY = tf.tensor2d(Float32Array.from(train.y), [train.n, 1])
    this.testX = tf.tensor2d(test.x, [test.n, 2])
    this.testY = tf.tensor2d(Float32Array.from(test.y), [test.n, 1])

    this.perm = new Int32Array(train.n)
    this.reshuffle()
  }

  private reshuffle() {
    const p = this.perm
    for (let i = 0; i < p.length; i++) p[i] = i
    for (let i = p.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1))
      const t = p[i]
      p[i] = p[j]
      p[j] = t
    }
    this.permPos = 0
  }

  private nextBatch(): Int32Array {
    const k = this.batchSize
    if (this.permPos + k > this.perm.length) this.reshuffle()
    const out = this.perm.slice(this.permPos, this.permPos + k)
    this.permPos += k
    return out
  }

  private activate(a: Activation, z: TFNS.Tensor): TFNS.Tensor {
    const tf = this.tf
    switch (a) {
      case 'relu':
        return tf.relu(z)
      case 'tanh':
        return tf.tanh(z)
      case 'sigmoid':
        return tf.sigmoid(z)
      case 'gelu': {
        const inner = z.add(z.mul(z).mul(z).mul(0.044715)).mul(Math.sqrt(2 / Math.PI))
        return z.mul(0.5).mul(tf.tanh(inner).add(1))
      }
    }
  }

  /** Forward pass. Returns the output logits and (optionally) every layer's activations. */
  private forward(x: TFNS.Tensor2D, acts?: TFNS.Tensor[]): TFNS.Tensor2D {
    let a: TFNS.Tensor2D = x
    const L = this.specs.length
    for (let l = 0; l < L - 1; l++) {
      const z = a.matMul(this.W[l] as TFNS.Tensor2D).add(this.b[l])
      a = this.activate(this.specs[l].activation, z) as TFNS.Tensor2D
      acts?.push(a)
    }
    const logits = a.matMul(this.W[L - 1] as TFNS.Tensor2D).add(this.b[L - 1]) as TFNS.Tensor2D
    acts?.push(this.tf.sigmoid(logits))
    return logits
  }

  /** Binary cross-entropy with logits: softplus(z) - y*z, averaged. */
  private bce(logits: TFNS.Tensor, y: TFNS.Tensor): TFNS.Scalar {
    const tf = this.tf
    return tf.softplus(logits).sub(y.mul(logits)).mean() as TFNS.Scalar
  }

  private loss(x: TFNS.Tensor2D, y: TFNS.Tensor2D): TFNS.Scalar {
    const tf = this.tf
    let total = this.bce(this.forward(x), y)
    if (this.l2 > 0) {
      let pen: TFNS.Tensor = tf.scalar(0)
      for (const W of this.W) pen = pen.add(W.square().sum())
      total = total.add(pen.mul(0.5 * this.l2)) as TFNS.Scalar
    }
    return total
  }

  /** One optimisation step on one mini-batch. */
  trainStep() {
    const tf = this.tf
    const idx = this.nextBatch()
    const t = this.step + 1
    tf.tidy(() => {
      const ib = tf.tensor1d(idx, 'int32')
      const xb = tf.gather(this.trainX, ib) as TFNS.Tensor2D
      const yb = tf.gather(this.trainY, ib) as TFNS.Tensor2D
      const { grads } = tf.variableGrads(() => this.loss(xb, yb), this.varList)
      for (const v of this.varList) this.applyUpdate(v, grads[v.name], t)
    })
    this.step = t
  }

  private applyUpdate(v: TFNS.Variable, g: TFNS.Tensor, t: number) {
    const tf = this.tf
    const lr = this.lr
    const kind = this.cfg.optimizer
    if (kind === 'sgd') {
      v.assign(v.sub(g.mul(lr)))
      return
    }
    const st = this.state.get(v.name) ?? {}
    if (kind === 'momentum') {
      const m = (st.m ? st.m.mul(0.9).add(g) : g.clone()) as TFNS.Tensor
      tf.keep(m)
      st.m?.dispose()
      st.m = m
      v.assign(v.sub(m.mul(lr)))
    } else {
      const b1 = 0.9
      const b2 = 0.999
      const m = (st.m ? st.m.mul(b1).add(g.mul(1 - b1)) : g.mul(1 - b1)) as TFNS.Tensor
      const vv = (
        st.v ? st.v.mul(b2).add(g.square().mul(1 - b2)) : g.square().mul(1 - b2)
      ) as TFNS.Tensor
      tf.keep(m)
      tf.keep(vv)
      st.m?.dispose()
      st.v?.dispose()
      st.m = m
      st.v = vv
      const mhat = m.div(1 - Math.pow(b1, t))
      const vhat = vv.div(1 - Math.pow(b2, t))
      v.assign(v.sub(mhat.div(vhat.sqrt().add(1e-8)).mul(lr)))
    }
    this.state.set(v.name, st)
  }

  /** Compute metrics, full-batch gradients, and per-layer statistics. */
  async snapshot(): Promise<Snapshot> {
    const tf = this.tf
    const L = this.specs.length
    const acts: TFNS.Tensor[] = []

    const out = tf.tidy(() => {
      const acc = (logits: TFNS.Tensor, y: TFNS.Tensor) =>
        logits.greater(0).equal(y.greater(0.5)).cast('float32').mean()
      const trLogits = this.forward(this.trainX, acts)
      acts.forEach((a) => tf.keep(a))
      const teLogits = this.forward(this.testX)
      const metrics = tf.stack([
        this.bce(trLogits, this.trainY),
        this.bce(teLogits, this.testY),
        acc(trLogits, this.trainY),
        acc(teLogits, this.testY),
      ])
      const { grads } = tf.variableGrads(() => this.loss(this.trainX, this.trainY), this.varList)
      const gW = this.W.map((w) => grads[w.name])
      return { metrics, gW }
    })

    const [metrics, ...rest] = await Promise.all([
      out.metrics.data(),
      ...out.gW.map((g) => g.data()),
      ...acts.map((a) => a.data()),
      ...this.W.map((w) => w.data()),
      ...this.b.map((b) => b.data()),
    ])
    const gData = rest.slice(0, L)
    const aData = rest.slice(L, 2 * L)
    const wData = rest.slice(2 * L, 3 * L)
    const bData = rest.slice(3 * L, 4 * L)
    out.metrics.dispose()
    out.gW.forEach((g) => g.dispose())
    acts.forEach((a) => a.dispose())

    const layers = this.specs.map((s, l) =>
      buildLayerStats({
        activation: s.activation,
        nout: s.nout,
        W: wData[l],
        gradW: gData[l],
        acts: aData[l],
        n: this.nTrain,
      }),
    )
    return {
      step: this.step,
      epoch: (this.step * this.batchSize) / this.nTrain,
      trainLoss: metrics[0],
      testLoss: metrics[1],
      trainAcc: metrics[2],
      testAcc: metrics[3],
      lr: this.lr,
      t: Date.now() - this.t0,
      weights: this.specs.map((_, l) => ({
        W: new Float32Array(wData[l]),
        b: new Float32Array(bData[l]),
      })),
      layers,
    }
  }

  dispose() {
    if (this.disposed) return
    this.disposed = true
    this.varList.forEach((v) => v.dispose())
    this.state.forEach((s) => {
      s.m?.dispose()
      s.v?.dispose()
    })
    this.trainX.dispose()
    this.trainY.dispose()
    this.testX.dispose()
    this.testY.dispose()
  }
}
