export type Activation = 'relu' | 'tanh' | 'sigmoid' | 'gelu'
export type OptimizerKind = 'sgd' | 'momentum' | 'adam'
export type DatasetKind = 'moons' | 'circles' | 'spiral' | 'xor' | 'blobs'

export interface HiddenLayer {
  units: number
  activation: Activation
}

export interface PlaygroundConfig {
  dataset: DatasetKind
  noise: number // 0..0.5
  samples: number // 100..1000
  testSplit: number // 0.1..0.5 fraction held out
  layers: HiddenLayer[]
  lr: number
  batchSize: number
  optimizer: OptimizerKind
  l2: number
  seed: number
}

export const DEFAULT_CONFIG: PlaygroundConfig = {
  dataset: 'spiral',
  noise: 0.08,
  samples: 400,
  testSplit: 0.3,
  layers: [
    { units: 8, activation: 'tanh' },
    { units: 8, activation: 'tanh' },
  ],
  lr: 0.03,
  batchSize: 32,
  optimizer: 'adam',
  l2: 0,
  seed: 7,
}

export interface Dataset {
  /** row-major [n][2] */
  x: Float32Array
  y: Uint8Array
  n: number
}

/** One dense layer of the network. The output layer is always 1 unit with a sigmoid. */
export interface LayerSpec {
  nin: number
  nout: number
  activation: Activation
}

export function layerSpecs(cfg: Pick<PlaygroundConfig, 'layers'>): LayerSpec[] {
  const specs: LayerSpec[] = []
  let nin = 2
  for (const l of cfg.layers) {
    specs.push({ nin, nout: l.units, activation: l.activation })
    nin = l.units
  }
  specs.push({ nin, nout: 1, activation: 'sigmoid' })
  return specs
}

export interface Hist {
  counts: number[]
  lo: number
  hi: number
}

export interface LayerStats {
  /** weights */
  wMean: number
  wStd: number
  wAbsMax: number
  wHist: Hist
  /** gradient of the loss wrt this layer's weights (full train batch) */
  gMeanAbs: number
  gNorm: number
  gStd: number
  gHist: Hist
  /** activations (after the activation function) on the train set */
  aMean: number
  aStd: number
  /** fraction of units that never fire on any sample (relu/gelu "dead") */
  deadPct: number
  /** fraction of activations that are exactly <= 0 (relu) */
  zeroPct: number
  /** fraction of activations stuck near the flat ends (tanh/sigmoid) */
  satPct: number
  aHist: Hist
  /** mean activation per unit, for colouring the network graph */
  unitMean: number[]
}

export interface Snapshot {
  step: number
  epoch: number
  trainLoss: number
  testLoss: number
  trainAcc: number
  testAcc: number
  /** the learning rate that was in effect when this snapshot was taken */
  lr: number
  /** time in ms since the run started */
  t: number
  weights: { W: Float32Array; b: Float32Array }[]
  layers: LayerStats[]
}

export function snapshotBytes(s: Snapshot): number {
  let bytes = 256
  for (const w of s.weights) bytes += w.W.byteLength + w.b.byteLength
  // each layer: 3 histograms (~24 numbers each) plus scalars and per-unit means
  for (const l of s.layers) bytes += 8 * (3 * 24 + 16 + l.unitMean.length)
  return bytes
}
