import type { Rule } from '../types'
import { avg, fmt, last } from '../util'

export const VANISH_RATIO = 0.05
export const VANISH_ABS = 1e-5
export const VANISH_MIN_LOSS = 0.2

/**
 * Gradients shrink layer by layer on their way back to the input: the first layers barely learn.
 * We compare the gradient size of the first and last layer, averaged over recent snapshots.
 */
export const vanishingGradients: Rule = ({ history, config }) => {
  if (config.layers.length < 2 || history.length < 5) return null
  const recent = last(history, 5)
  const tail = recent[recent.length - 1]
  if (tail.step < 20 || tail.trainLoss < VANISH_MIN_LOSS) return null // converged nets have small gradients too

  const nL = tail.layers.length
  const first = avg(recent.map((s) => s.layers[0].gNorm))
  const lastL = avg(recent.map((s) => s.layers[nL - 1].gNorm))
  if (!(lastL > 0)) return null
  const ratio = first / lastL
  if (!(ratio < VANISH_RATIO || first < VANISH_ABS)) return null

  const squashing = config.layers.some((l) => l.activation === 'sigmoid' || l.activation === 'tanh')
  return {
    id: 'vanishing',
    severity: ratio < 0.01 || first < VANISH_ABS ? 'critical' : 'warn',
    title: 'Vanishing gradients',
    explanation: `The first layer's gradient is ${fmt(ratio * 100, 2)}% of the last layer's. Each layer multiplies the gradient by a number below 1 (sigmoid/tanh slopes are at most 1 and 0.25), so by the time it reaches the early layers almost nothing is left and they stop learning.`,
    evidence: {
      step: tail.step,
      chart: 'grad',
      detail: `layer 1 gradient ${fmt(first)} vs output layer ${fmt(lastL)}`,
    },
    fix: squashing
      ? {
          label: 'Switch all hidden layers to ReLU',
          patch: { layers: config.layers.map((l) => ({ ...l, activation: 'relu' as const })) },
          reset: true,
        }
      : undefined,
  }
}
