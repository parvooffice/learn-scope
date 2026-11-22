import type { Rule } from '../types'
import { avg, fmt, last, niceLr } from '../util'

export const LR_HIGH_WINDOW = 20
export const LR_HIGH_UP_FRACTION = 0.3
export const LR_HIGH_RISE = 1.3

/**
 * Training loss going UP, or zig-zagging, means the optimizer keeps overshooting the minimum.
 * (Exploding gradients are handled by their own rule; this one catches the milder cases.)
 */
export const learningRateTooHigh: Rule = ({ history, config }) => {
  if (history.length < 12) return null
  const win = last(history, LR_HIGH_WINDOW)
  const losses = win.map((s) => s.trainLoss)
  if (losses.some((x) => !Number.isFinite(x))) return null

  let ups = 0
  let worstJump = 0
  let worstStep = win[win.length - 1].step
  for (let i = 1; i < losses.length; i++) {
    if (losses[i] > losses[i - 1] * 1.05) {
      ups++
      const jump = losses[i] - losses[i - 1]
      if (jump > worstJump) {
        worstJump = jump
        worstStep = win[i].step
      }
    }
  }
  const upFraction = ups / (losses.length - 1)
  const best = Math.min(...history.map((s) => s.trainLoss))
  const nowAvg = avg(last(losses, 5))
  const rising = nowAvg > best * LR_HIGH_RISE && nowAvg - best > 0.05

  if (!(upFraction >= LR_HIGH_UP_FRACTION || rising)) return null
  const severe = nowAvg > best * 2 && nowAvg - best > 0.2
  return {
    id: 'lr-high',
    severity: severe ? 'critical' : 'warn',
    title: 'Learning rate looks too high',
    explanation: `The training loss keeps jumping back up (${ups} rises in the last ${losses.length - 1} snapshots; best ${fmt(best)}, now ${fmt(nowAvg)}). With steps that are too big the optimizer leaps over the bottom of the valley and lands on the other side, again and again.`,
    evidence: {
      step: worstStep,
      chart: 'loss',
      detail: `loss jumped by ${fmt(worstJump)} here`,
    },
    fix: {
      label: `Lower the learning rate to ${niceLr(config.lr / 3)}`,
      patch: { lr: niceLr(config.lr / 3) },
      reset: false,
    },
  }
}
