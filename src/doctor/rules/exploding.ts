import type { Rule } from '../types'
import { fmt, niceLr, snapshotHasNaN } from '../util'

export const EXPLODING_GRAD_NORM = 100
export const EXPLODING_LOSS = 20

/** Gradients or loss have blown up (or became NaN/Infinity). */
export const explodingGradients: Rule = ({ history, config }) => {
  if (history.length < 2) return null
  const hit = history.find(
    (s) =>
      snapshotHasNaN(s) ||
      s.trainLoss > EXPLODING_LOSS ||
      s.layers.some((l) => l.gNorm > EXPLODING_GRAD_NORM),
  )
  if (!hit) return null
  const broken = snapshotHasNaN(hit)
  const worst = Math.max(...hit.layers.map((l) => (Number.isFinite(l.gNorm) ? l.gNorm : Infinity)))
  return {
    id: 'exploding',
    severity: 'critical',
    title: broken ? 'Training produced NaN / Infinity' : 'Exploding gradients',
    explanation: broken
      ? 'The numbers overflowed, so the weights are now NaN and the network cannot recover. This usually means the learning rate was too large for this network.'
      : `The gradient (the "push" on the weights) became huge (norm ${fmt(worst)}). Giant steps overshoot the valley of the loss surface and make training unstable. Lowering the learning rate shrinks every step.`,
    evidence: {
      step: hit.step,
      chart: 'grad',
      detail: broken ? 'first step with non-finite values' : `gradient norm ${fmt(worst)} at this step`,
    },
    fix: {
      label: `Lower the learning rate to ${niceLr(config.lr / 10)} and restart`,
      patch: { lr: niceLr(config.lr / 10) },
      reset: true,
    },
  }
}
