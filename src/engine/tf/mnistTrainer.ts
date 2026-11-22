import type * as TFNS from '@tensorflow/tfjs'
import type {
  MapStack,
  MnistConfig,
  MnistEval,
  MnistFocus,
  MnistSnapshot,
} from '../../worker/mnistProtocol'
import { mulberry32 } from '../../lib/rng'

type TF = typeof TFNS
const PIX = 784

export interface MnistDataset {
  images: Uint8Array
  labels: Uint8Array
  nTrain: number
  nTest: number
}

/** A small CNN or MLP on MNIST, trained with TensorFlow.js (tf.layers) in a hand-written loop. */
export class MnistTrainer {
  step = 0
  readonly params: number
  /** milliseconds actually spent training (excludes pauses), for the img/s benchmark */
  trainMs = 0

  private readonly tf: TF
  private readonly cfg: MnistConfig
  private readonly model: TFNS.Sequential
  private readonly feat: TFNS.LayersModel
  private readonly trainX: TFNS.Tensor4D
  private readonly trainY: TFNS.Tensor2D
  private readonly testX: TFNS.Tensor4D
  private readonly testLabels: Uint8Array
  private readonly nTrain: number
  private readonly nTest: number
  private perm: Int32Array
  private permPos = 0
  private readonly rng: () => number
  private lossAcc: { loss: number; acc: number; n: number } = { loss: 0, acc: 0, n: 0 }
  private lastLoss = 2.3
  private lastAcc = 0.1
  private focusPixels: Float32Array | null = null

  constructor(tf: TF, cfg: MnistConfig, data: MnistDataset) {
    this.tf = tf
    this.cfg = cfg
    this.nTrain = data.nTrain
    this.nTest = data.nTest
    this.rng = mulberry32(cfg.seed * 7 + 3)

    const L = tf.layers
    const model = tf.sequential()
    if (cfg.model === 'cnn') {
      model.add(
        L.conv2d({
          inputShape: [28, 28, 1],
          filters: 8,
          kernelSize: 5,
          activation: 'relu',
          name: 'conv1',
          kernelInitializer: 'heNormal',
        }),
      )
      model.add(L.maxPooling2d({ poolSize: 2, name: 'pool1' }))
      model.add(
        L.conv2d({
          filters: 16,
          kernelSize: 3,
          activation: 'relu',
          name: 'conv2',
          kernelInitializer: 'heNormal',
        }),
      )
      model.add(L.maxPooling2d({ poolSize: 2, name: 'pool2' }))
      model.add(L.flatten({ name: 'flat' }))
      model.add(L.dense({ units: 10, activation: 'softmax', name: 'out' }))
    } else {
      model.add(L.flatten({ inputShape: [28, 28, 1], name: 'flat' }))
      model.add(
        L.dense({ units: 128, activation: 'relu', name: 'fc1', kernelInitializer: 'heNormal' }),
      )
      model.add(
        L.dense({ units: 64, activation: 'relu', name: 'fc2', kernelInitializer: 'heNormal' }),
      )
      model.add(L.dense({ units: 10, activation: 'softmax', name: 'out' }))
    }
    model.compile({
      optimizer: tf.train.adam(cfg.lr),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    })
    this.model = model
    this.params = model.countParams()

    const outs =
      cfg.model === 'cnn'
        ? [model.getLayer('conv1').output, model.getLayer('conv2').output, model.output]
        : [model.getLayer('fc1').output, model.output]
    this.feat = tf.model({ inputs: model.inputs, outputs: outs as TFNS.SymbolicTensor[] })

    const f32 = (from: number, count: number) => {
      const a = new Float32Array(count * PIX)
      for (let i = 0; i < a.length; i++) a[i] = data.images[from * PIX + i] / 255
      return a
    }
    this.trainX = tf.tensor4d(f32(0, data.nTrain), [data.nTrain, 28, 28, 1])
    const y = new Float32Array(data.nTrain * 10)
    for (let i = 0; i < data.nTrain; i++) y[i * 10 + data.labels[i]] = 1
    this.trainY = tf.tensor2d(y, [data.nTrain, 10])
    this.testX = tf.tensor4d(f32(data.nTrain, data.nTest), [data.nTest, 28, 28, 1])
    this.testLabels = data.labels.slice(data.nTrain, data.nTrain + data.nTest)

    this.perm = new Int32Array(data.nTrain)
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

  async trainStep() {
    const tf = this.tf
    const k = this.cfg.batchSize
    if (this.permPos + k > this.perm.length) this.reshuffle()
    const idx = this.perm.slice(this.permPos, this.permPos + k)
    this.permPos += k
    const ib = tf.tensor1d(idx, 'int32')
    const xb = tf.gather(this.trainX, ib) as TFNS.Tensor4D
    const yb = tf.gather(this.trainY, ib) as TFNS.Tensor2D
    const t0 = performance.now()
    const r = (await this.model.trainOnBatch(xb, yb)) as number[]
    this.trainMs += performance.now() - t0
    tf.dispose([ib, xb, yb])
    this.step++
    this.lossAcc.loss += r[0]
    this.lossAcc.acc += r[1]
    this.lossAcc.n++
  }

  /** Feature maps and class probabilities for one image (pixels 0..1). */
  async focus(pixels: Float32Array): Promise<MnistFocus> {
    const tf = this.tf
    const x = tf.tensor4d(pixels, [1, 28, 28, 1])
    const outs = this.feat.predict(x) as TFNS.Tensor[]
    const toStack = async (t: TFNS.Tensor): Promise<MapStack> => {
      // [1,h,w,c] -> [c,h,w]
      const s = t.shape
      const chw = tf.transpose(t.squeeze([0]), [2, 0, 1])
      const data = (await chw.data()) as Float32Array
      chw.dispose()
      return { count: s[3]!, size: s[1]!, data: new Float32Array(data) }
    }
    const out: MnistFocus = { probs: [] }
    if (this.cfg.model === 'cnn') {
      out.conv1 = await toStack(outs[0])
      out.conv2 = await toStack(outs[1])
      out.probs = Array.from((await outs[2].data()) as Float32Array)
    } else {
      out.hidden = Array.from((await outs[0].data()) as Float32Array)
      out.probs = Array.from((await outs[1].data()) as Float32Array)
    }
    tf.dispose([x, ...outs])
    return out
  }

  setFocus(pixels: Float32Array | null) {
    this.focusPixels = pixels
  }

  private async filters(): Promise<MapStack> {
    const tf = this.tf
    if (this.cfg.model === 'cnn') {
      const [k] = this.model.getLayer('conv1').getWeights() // [5,5,1,8]
      const t = tf.transpose(k.reshape([5, 5, 8]), [2, 0, 1])
      const data = (await t.data()) as Float32Array
      t.dispose()
      return { count: 8, size: 5, data: new Float32Array(data) }
    }
    const [k] = this.model.getLayer('fc1').getWeights() // [784,128]
    const first = k.slice([0, 0], [784, 16])
    const t = tf.transpose(first) // [16,784]
    const data = (await t.data()) as Float32Array
    tf.dispose([first, t])
    return { count: 16, size: 28, data: new Float32Array(data) }
  }

  /** Accuracy, loss, confusion matrix and the most confidently wrong test images. */
  async evaluate(): Promise<MnistEval> {
    const tf = this.tf
    const confusion = new Array<number>(100).fill(0)
    const wrong: { index: number; pred: number; truth: number; conf: number }[] = []
    let loss = 0
    let correct = 0
    const CH = 500
    for (let s = 0; s < this.nTest; s += CH) {
      const n = Math.min(CH, this.nTest - s)
      const probs = tf.tidy(
        () => this.model.predict(this.testX.slice([s, 0, 0, 0], [n, 28, 28, 1])) as TFNS.Tensor,
      )
      const p = (await probs.data()) as Float32Array
      probs.dispose()
      for (let i = 0; i < n; i++) {
        const truth = this.testLabels[s + i]
        let best = 0
        for (let c = 1; c < 10; c++) if (p[i * 10 + c] > p[i * 10 + best]) best = c
        confusion[truth * 10 + best]++
        loss += -Math.log(Math.max(1e-7, p[i * 10 + truth]))
        if (best === truth) correct++
        else wrong.push({ index: s + i, pred: best, truth, conf: p[i * 10 + best] })
      }
    }
    wrong.sort((a, b) => b.conf - a.conf)
    return {
      testAcc: correct / this.nTest,
      testLoss: loss / this.nTest,
      confusion,
      worst: wrong.slice(0, 12),
    }
  }

  async snapshot(withEval: boolean): Promise<MnistSnapshot> {
    const n = Math.max(1, this.lossAcc.n)
    if (this.lossAcc.n > 0) {
      this.lastLoss = this.lossAcc.loss / n
      this.lastAcc = this.lossAcc.acc / n
    }
    this.lossAcc = { loss: 0, acc: 0, n: 0 }
    const elapsed = this.trainMs / 1000
    return {
      step: this.step,
      epoch: (this.step * this.cfg.batchSize) / this.nTrain,
      trainLoss: this.lastLoss,
      trainAcc: this.lastAcc,
      imgPerSec: elapsed > 0 ? (this.step * this.cfg.batchSize) / elapsed : 0,
      elapsed,
      filters: await this.filters(),
      eval: withEval ? await this.evaluate() : undefined,
      focus: this.focusPixels ? await this.focus(this.focusPixels) : undefined,
    }
  }

  dispose() {
    this.model.dispose() // also frees the layers shared with the feature-map model
    this.trainX.dispose()
    this.trainY.dispose()
    this.testX.dispose()
  }
}
