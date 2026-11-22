import type { Rule } from '../types'
import { fmt, last, niceLr } from '../util'

export const LR_LOW_MIN_STEPS = 150
export const LR_LOW_WINDOW = 30
export const LR_LOW_IMPROVEMENT = 0.01
export const LR_LOW_LOSS = 0.45

/** Loss is high but has hardly moved for a long time: the steps are too small to make progress. */
export const learningRateTooLow: Rule = ({ history, config }) => {
  const tail = history[history.length - 1]
  if (!tail || history.length < LR_LOW_WINDOW || tail.step < LR_LOW_MIN_STEPS) return null
  const win = last(history, LR_LOW_WINDOW)
  const startLoss = win[0].trainLoss
  const endLoss = win[win.length - 1].trainLoss
  if (!Number.isFinite(startLoss) || !Number.isFinite(endLoss)) return null
  if (win[win.length - 1].step - win[0].step < 60) return null
  if (endLoss < LR_LOW_LOSS || tail.trainAcc > 0.95) return null
  const improvement = (startLoss - endLoss) / Math.max(startLoss, 1e-9)
  if (improvement >= LR_LOW_IMPROVEMENT) return null

  return {
    id: 'lr-low',
    severity: 'warn',
    title: 'Learning rate looks too low (or training is stuck)',
    explanation: `Over the last ${win[win.length - 1].step - win[0].step} steps the loss only changed by ${fmt(improvement * 100, 2)}%, and it is still high (${fmt(endLoss)}). Tiny steps crawl across a flat region. Try a bigger learning rate; if that doesn't help, the model may need more capacity.`,
    evidence: {
      step: win[0].step,
      chart: 'loss',
      detail: `loss ${fmt(startLoss)} → ${fmt(endLoss)} over this stretch`,
    },
    fix: {
      label: `Raise the learning rate to ${niceLr(config.lr * 3)}`,
      patch: { lr: niceLr(config.lr * 3) },
      reset: false,
    },
  }
}
