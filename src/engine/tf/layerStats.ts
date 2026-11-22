import type { Activation, LayerStats } from '../types'
import { absMax, histogram, l2Norm, meanAbs, meanStd } from '../../lib/stats'

/**
 * Turn raw numbers (weights, gradients, activations) of one layer into the compact statistics
 * we store in each snapshot. Pure function: no TensorFlow here, so it is easy to unit-test.
 *
 * @param acts  activations laid out [sample][unit], length n * nout
 */
export function buildLayerStats(args: {
  activation: Activation
  nout: number
  W: ArrayLike<number>
  gradW: ArrayLike<number>
  acts: ArrayLike<number>
  n: number
}): LayerStats {
  const { activation, nout, W, gradW, acts, n } = args
  const w = meanStd(W)
  const wMax = absMax(W)
  const g = meanStd(gradW)
  const gMax = absMax(gradW)
  const a = meanStd(acts)

  // per-unit stats
  const unitMean = new Array<number>(nout).fill(0)
  const unitMax = new Array<number>(nout).fill(-Infinity)
  let zero = 0
  let sat = 0
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < nout; j++) {
      const v = acts[i * nout + j]
      unitMean[j] += v
      if (v > unitMax[j]) unitMax[j] = v
      if (v <= 0) zero++
      if (activation === 'tanh' && Math.abs(v) > 0.95) sat++
      if (activation === 'sigmoid' && (v < 0.05 || v > 0.95)) sat++
    }
  }
  for (let j = 0; j < nout; j++) unitMean[j] /= Math.max(1, n)

  let dead = 0
  if (activation === 'relu' || activation === 'gelu') {
    for (let j = 0; j < nout; j++) if (unitMax[j] < 1e-3) dead++
  }
  const total = Math.max(1, n * nout)

  let aHist
  if (activation === 'tanh') aHist = histogram(acts, 24, -1, 1)
  else if (activation === 'sigmoid') aHist = histogram(acts, 24, 0, 1)
  else {
    let lo = 0
    let hi = 0
    for (let i = 0; i < acts.length; i++) {
      lo = Math.min(lo, acts[i])
      hi = Math.max(hi, acts[i])
    }
    aHist = histogram(acts, 24, lo, hi <= lo ? lo + 1 : hi)
  }

  return {
    wMean: w.mean,
    wStd: w.std,
    wAbsMax: wMax,
    wHist: histogram(W, 24, -(wMax || 1), wMax || 1),
    gMeanAbs: meanAbs(gradW),
    gNorm: l2Norm(gradW),
    gStd: g.std,
    gHist: histogram(gradW, 24, -(gMax || 1e-12), gMax || 1e-12),
    aMean: a.mean,
    aStd: a.std,
    deadPct: dead / Math.max(1, nout),
    zeroPct: zero / total,
    satPct: sat / total,
    aHist,
    unitMean,
  }
}
