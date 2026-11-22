import {
  DEFAULT_CONFIG,
  type LayerStats,
  type PlaygroundConfig,
  type Snapshot,
} from '../engine/types'

export const emptyHist = () => ({ counts: new Array(24).fill(0), lo: 0, hi: 1 })

export function mkLayer(over: Partial<LayerStats> = {}): LayerStats {
  return {
    wMean: 0,
    wStd: 0.5,
    wAbsMax: 1,
    wHist: emptyHist(),
    gMeanAbs: 0.05,
    gNorm: 0.2,
    gStd: 0.05,
    gHist: emptyHist(),
    aMean: 0.1,
    aStd: 0.4,
    deadPct: 0,
    zeroPct: 0.3,
    satPct: 0.05,
    aHist: emptyHist(),
    unitMean: [0.1],
    ...over,
  }
}

export function mkSnap(
  step: number,
  cfg: PlaygroundConfig,
  over: Partial<Snapshot> & { layerOver?: (i: number) => Partial<LayerStats> } = {},
): Snapshot {
  const { layerOver, ...rest } = over
  const n = cfg.layers.length + 1
  return {
    step,
    epoch: step / 10,
    trainLoss: 0.5,
    testLoss: 0.5,
    trainAcc: 0.8,
    testAcc: 0.8,
    lr: cfg.lr,
    t: step * 10,
    weights: Array.from({ length: n }, () => ({ W: new Float32Array(4), b: new Float32Array(2) })),
    layers: Array.from({ length: n }, (_, i) => mkLayer(layerOver?.(i))),
    ...rest,
  }
}

/** Build a history of `count` snapshots (step = 5 * index) from a per-index factory. */
export function mkHistory(
  cfg: PlaygroundConfig,
  count: number,
  f: (i: number) => Partial<Snapshot> & { layerOver?: (layer: number) => Partial<LayerStats> },
): Snapshot[] {
  return Array.from({ length: count }, (_, i) => mkSnap((i + 1) * 5, cfg, f(i)))
}

export const cfgWith = (over: Partial<PlaygroundConfig>): PlaygroundConfig => ({
  ...DEFAULT_CONFIG,
  ...over,
})
