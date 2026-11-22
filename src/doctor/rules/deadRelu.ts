import type { Rule } from '../types'
import { fmt } from '../util'

export const DEAD_WARN = 0.5
export const DEAD_CRITICAL = 0.8

/** ReLU units that output exactly 0 for every training sample: they get no gradient and never recover. */
export const deadRelus: Rule = ({ history, config }) => {
  if (history.length < 3) return null
  const tail = history[history.length - 1]
  let worstLayer = -1
  let worst = 0
  config.layers.forEach((l, i) => {
    if ((l.activation !== 'relu' && l.activation !== 'gelu') || l.units < 2) return
    const dead = tail.layers[i]?.deadPct ?? 0
    if (dead > worst) {
      worst = dead
      worstLayer = i
    }
  })
  if (worstLayer < 0 || worst <= DEAD_WARN) return null

  // evidence: first step at which that layer crossed the threshold
  const first = history.find((s) => (s.layers[worstLayer]?.deadPct ?? 0) > DEAD_WARN) ?? tail
  return {
    id: 'dead-relu',
    severity: worst > DEAD_CRITICAL ? 'critical' : 'warn',
    title: 'Dead ReLU units',
    explanation: `${fmt(worst * 100, 3)}% of the units in hidden layer ${worstLayer + 1} output 0 for every sample. A ReLU outputs 0 for negative inputs and passes no gradient there, so a "dead" unit can never wake up again. It is often caused by a learning rate that kicked the weights too far.`,
    evidence: {
      step: first.step,
      chart: 'activations',
      detail: `layer ${worstLayer + 1}: ${fmt(worst * 100, 3)}% dead`,
    },
    fix: {
      label: 'Switch ReLU layers to GELU (smooth, no hard zero) and restart',
      patch: {
        layers: config.layers.map((l) => (l.activation === 'relu' ? { ...l, activation: 'gelu' as const } : l)),
      },
      reset: true,
    },
  }
}
