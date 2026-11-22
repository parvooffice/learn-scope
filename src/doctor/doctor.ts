import type { Diagnosis, DoctorContext, Rule, Severity } from './types'
import { explodingGradients } from './rules/exploding'
import { vanishingGradients } from './rules/vanishing'
import { deadRelus } from './rules/deadRelu'
import { learningRateTooHigh } from './rules/lrHigh'
import { learningRateTooLow } from './rules/lrLow'
import { overfitting } from './rules/overfitting'
import { saturatedUnits } from './rules/saturation'

export const RULES: Rule[] = [
  explodingGradients,
  vanishingGradients,
  deadRelus,
  saturatedUnits,
  learningRateTooHigh,
  learningRateTooLow,
  overfitting,
]

const ORDER: Record<Severity, number> = { critical: 0, warn: 1, info: 2 }

/** Run every rule over the training history. Deterministic: same input, same output. */
export function diagnose(ctx: DoctorContext): Diagnosis[] {
  const out: Diagnosis[] = []
  for (const rule of RULES) {
    const d = rule(ctx)
    if (d) out.push(d)
  }
  out.sort((a, b) => ORDER[a.severity] - ORDER[b.severity])

  // if the run just exploded, the other symptoms are consequences: show the root cause first only
  if (out.some((d) => d.id === 'exploding')) return out.filter((d) => d.id === 'exploding')

  const tail = ctx.history[ctx.history.length - 1]
  if (out.length === 0 && tail && tail.step >= 30) {
    out.push({
      id: 'healthy',
      severity: 'info',
      title: 'No problems detected',
      explanation:
        'Gradients flow through all layers, units are active, and training and test loss move together. Keep watching: some problems (like overfitting) only show up later.',
      evidence: { step: tail.step, chart: 'loss', detail: 'all checks passed at this step' },
    })
  }
  return out
}
