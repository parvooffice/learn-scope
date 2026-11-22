import type { Rule } from '../types'
import { fmt } from '../util'

export const SAT_WARN = 0.6
export const SAT_CRITICAL = 0.85
export const SAT_MIN_LOSS = 0.25

/** tanh/sigmoid units pinned at their flat ends have almost zero slope, so they barely learn. */
export const saturatedUnits: Rule = ({ history, config }) => {
  if (history.length < 3) return null
  const tail = history[history.length - 1]
  // A network that has already learned the task is allowed to use saturated units (a spiral
  // boundary needs sharp turns). Only flag saturation while the loss is still high.
  if (tail.trainLoss < SAT_MIN_LOSS) return null
  let worstLayer = -1
  let worst = 0
  config.layers.forEach((l, i) => {
    if (l.activation !== 'tanh' && l.activation !== 'sigmoid') return
    const sat = tail.layers[i]?.satPct ?? 0
    if (sat > worst) {
      worst = sat
      worstLayer = i
    }
  })
  if (worstLayer < 0 || worst <= SAT_WARN) return null
  const first = history.find((s) => (s.layers[worstLayer]?.satPct ?? 0) > SAT_WARN) ?? tail
  const act = config.layers[worstLayer].activation
  return {
    id: 'saturated',
    severity: worst > SAT_CRITICAL ? 'critical' : 'warn',
    title: `Saturated ${act} units`,
    explanation: `${fmt(worst * 100, 3)}% of the ${act} outputs in hidden layer ${worstLayer + 1} sit at the flat ends of the curve. There the slope is almost 0, so gradients through them nearly vanish. Weights that grew too large (or a learning rate that was too high) push units into this regime.`,
    evidence: {
      step: first.step,
      chart: 'activations',
      detail: `layer ${worstLayer + 1}: ${fmt(worst * 100, 3)}% saturated`,
    },
    fix: {
      label: 'Switch saturated layers to ReLU and restart',
      patch: {
        layers: config.layers.map((l) =>
          l.activation === 'tanh' || l.activation === 'sigmoid'
            ? { ...l, activation: 'relu' as const }
            : l,
        ),
      },
      reset: true,
    },
  }
}
