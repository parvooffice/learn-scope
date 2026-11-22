import type { Rule } from '../types'
import { avg, fmt, last } from '../util'

export const OVERFIT_GAP = 0.15
export const OVERFIT_RISE = 1.1
export const OVERFIT_ACC_GAP = 0.05

/** Training loss keeps improving but test loss has turned around: the model is memorising the training set. */
export const overfitting: Rule = ({ history, config }) => {
  if (history.length < 15) return null
  const win = last(history, 10)
  const gap = avg(win.map((s) => s.testLoss - s.trainLoss))
  const accGap = avg(win.map((s) => s.trainAcc - s.testAcc))
  const nowTest = avg(win.map((s) => s.testLoss))
  let best = history[0]
  for (const s of history) if (s.testLoss < best.testLoss) best = s
  if (!Number.isFinite(gap)) return null
  if (gap < OVERFIT_GAP || accGap < OVERFIT_ACC_GAP) return null
  if (nowTest < best.testLoss * OVERFIT_RISE) return null

  const needL2 = config.l2 < 0.005
  return {
    id: 'overfitting',
    severity: gap > 0.4 ? 'critical' : 'warn',
    title: 'Overfitting',
    explanation: `Test loss (${fmt(nowTest)}) is ${fmt(gap)} above training loss and has risen since its best value of ${fmt(best.testLoss)}. The network is memorising training points instead of learning the pattern, so it does worse on data it hasn't seen.`,
    evidence: {
      step: best.step,
      chart: 'loss',
      detail: `test loss was lowest here (${fmt(best.testLoss)}); it gets worse afterwards`,
    },
    fix: needL2
      ? { label: 'Add L2 regularization (0.01)', patch: { l2: 0.01 }, reset: false }
      : {
          label: 'Use a smaller network (halve the units per layer) and restart',
          patch: {
            layers: config.layers.map((l) => ({
              ...l,
              units: Math.max(2, Math.round(l.units / 2)),
            })),
          },
          reset: true,
        },
  }
}
