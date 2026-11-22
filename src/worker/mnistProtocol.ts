export type MnistModelKind = 'mlp' | 'cnn'

export interface MnistConfig {
  model: MnistModelKind
  lr: number
  batchSize: number
  seed: number
}

export const DEFAULT_MNIST: MnistConfig = { model: 'cnn', lr: 0.003, batchSize: 64, seed: 1 }

export type ToMnistWorker =
  | { type: 'data'; images: Uint8Array; labels: Uint8Array; nTrain: number; nTest: number }
  | { type: 'configure'; config: MnistConfig }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'step'; n: number }
  | { type: 'setSpeed'; stepsPerTick: number }
  /** which image to visualise: pixels in 0..1 (28×28) */
  | { type: 'focus'; pixels: Float32Array }

/** A set of small images: `count` maps of size×size packed one after another. */
export interface MapStack {
  count: number
  size: number
  data: Float32Array
}

export interface MnistEval {
  testAcc: number
  testLoss: number
  /** confusion[trueClass * 10 + predictedClass] */
  confusion: number[]
  /** the test images the model is most confidently wrong about */
  worst: { index: number; pred: number; truth: number; conf: number }[]
}

export interface MnistFocus {
  probs: number[]
  /** feature maps for the focused image (CNN: conv1 + conv2; MLP: hidden activations as 1-px maps) */
  conv1?: MapStack
  conv2?: MapStack
  hidden?: number[]
}

export interface MnistSnapshot {
  step: number
  epoch: number
  trainLoss: number
  trainAcc: number
  /** images processed per second since the run started (for the performance budget) */
  imgPerSec: number
  /** seconds since the run started, training only */
  elapsed: number
  /** first-layer filters: CNN conv1 kernels (k×k) or MLP first-layer weights as 28×28 images */
  filters: MapStack
  eval?: MnistEval
  focus?: MnistFocus
}

export type FromMnistWorker =
  | { type: 'ready'; backend: string; notes: string[] }
  | { type: 'built'; model: MnistModelKind; params: number }
  | { type: 'snapshot'; snapshot: MnistSnapshot }
  | { type: 'focus'; focus: MnistFocus }
  | { type: 'status'; running: boolean }
  | { type: 'error'; message: string }
