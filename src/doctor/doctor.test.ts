import { describe, expect, it } from 'vitest'
import { diagnose } from './doctor'
import { explodingGradients } from './rules/exploding'
import { vanishingGradients } from './rules/vanishing'
import { deadRelus } from './rules/deadRelu'
import { learningRateTooHigh } from './rules/lrHigh'
import { learningRateTooLow } from './rules/lrLow'
import { overfitting } from './rules/overfitting'
import { saturatedUnits } from './rules/saturation'
import { cfgWith, mkHistory } from './testUtils'

const reluCfg = cfgWith({
  layers: [
    { units: 8, activation: 'relu' },
    { units: 8, activation: 'relu' },
    { units: 8, activation: 'relu' },
  ],
})
const sigmoidCfg = cfgWith({
  layers: [
    { units: 8, activation: 'sigmoid' },
    { units: 8, activation: 'sigmoid' },
    { units: 8, activation: 'sigmoid' },
  ],
})
const tanhCfg = cfgWith({
  layers: [
    { units: 8, activation: 'tanh' },
    { units: 8, activation: 'tanh' },
  ],
})

/** A boring, healthy run: loss falls smoothly, test tracks train. */
const healthy = (cfg = reluCfg, n = 40) =>
  mkHistory(cfg, n, (i) => ({
    trainLoss: 0.7 * Math.exp(-i / 12) + 0.08,
    testLoss: 0.7 * Math.exp(-i / 12) + 0.1,
    trainAcc: 0.98,
    testAcc: 0.96,
  }))

describe('Model Doctor: healthy runs are not flagged', () => {
  it('no rule fires and the doctor says so', () => {
    const history = healthy()
    const d = diagnose({ history, config: reluCfg })
    expect(d.map((x) => x.id)).toEqual(['healthy'])
    expect(d[0].severity).toBe('info')
  })
  it('stays quiet for a very short history', () => {
    const d = diagnose({ history: healthy(reluCfg, 2), config: reluCfg })
    expect(d).toEqual([])
  })
})

describe('exploding gradients', () => {
  it('fires on a huge gradient and suggests lr/10 with restart', () => {
    const history = mkHistory(reluCfg, 10, (i) => ({
      layerOver: (l) => (i >= 6 && l === 0 ? { gNorm: 500 } : {}),
    }))
    const d = explodingGradients({ history, config: reluCfg })!
    expect(d.severity).toBe('critical')
    expect(d.evidence.step).toBe(history[6].step)
    expect(d.evidence.chart).toBe('grad')
    expect(d.fix?.patch.lr).toBeCloseTo(reluCfg.lr / 10, 5)
    expect(d.fix?.reset).toBe(true)
  })
  it('fires on NaN loss', () => {
    const history = mkHistory(reluCfg, 5, (i) => (i === 3 ? { trainLoss: NaN } : {}))
    expect(explodingGradients({ history, config: reluCfg })?.title).toMatch(/NaN/)
  })
  it('does not fire on normal gradients', () => {
    expect(explodingGradients({ history: healthy(), config: reluCfg })).toBeNull()
  })
  it('hides downstream symptoms when exploded', () => {
    const history = mkHistory(sigmoidCfg, 30, (i) => ({
      trainLoss: i > 20 ? 50 : 0.7,
      layerOver: () => ({ gNorm: i > 20 ? 1e4 : 1e-6 }),
    }))
    expect(diagnose({ history, config: sigmoidCfg }).map((d) => d.id)).toEqual(['exploding'])
  })
})

describe('vanishing gradients', () => {
  const vanishing = mkHistory(sigmoidCfg, 30, () => ({
    trainLoss: 0.69,
    layerOver: (l) => ({ gNorm: [1e-6, 1e-5, 1e-4, 0.3][l] }),
  }))
  it('fires when early layers get a tiny fraction of the gradient', () => {
    const d = vanishingGradients({ history: vanishing, config: sigmoidCfg })!
    expect(d.severity).toBe('critical')
    expect(d.evidence.chart).toBe('grad')
    expect(d.fix?.label).toMatch(/ReLU/)
    expect(d.fix?.patch.layers?.every((l) => l.activation === 'relu')).toBe(true)
  })
  it('does not fire when the gradient is balanced', () => {
    expect(vanishingGradients({ history: healthy(), config: reluCfg })).toBeNull()
  })
  it('does not fire once the loss is already low (converged)', () => {
    const converged = mkHistory(sigmoidCfg, 30, () => ({
      trainLoss: 0.02,
      layerOver: (l) => ({ gNorm: [1e-6, 1e-5, 1e-4, 0.3][l] }),
    }))
    expect(vanishingGradients({ history: converged, config: sigmoidCfg })).toBeNull()
  })
  it('does not fire for a single hidden layer', () => {
    const one = cfgWith({ layers: [{ units: 4, activation: 'sigmoid' }] })
    const h = mkHistory(one, 30, () => ({
      trainLoss: 0.69,
      layerOver: (l) => ({ gNorm: [1e-7, 0.3][l] }),
    }))
    expect(vanishingGradients({ history: h, config: one })).toBeNull()
  })
})

describe('dead ReLUs', () => {
  it('fires when more than half of a layer is dead, pointing at the first crossing', () => {
    const history = mkHistory(reluCfg, 20, (i) => ({
      layerOver: (l) => (l === 1 ? { deadPct: i < 8 ? 0.1 : 0.75 } : {}),
    }))
    const d = deadRelus({ history, config: reluCfg })!
    expect(d.severity).toBe('warn')
    expect(d.evidence.step).toBe(history[8].step)
    expect(d.evidence.chart).toBe('activations')
    expect(d.explanation).toMatch(/layer 2/)
  })
  it('is critical above 80%', () => {
    const history = mkHistory(reluCfg, 10, () => ({ layerOver: () => ({ deadPct: 0.9 }) }))
    expect(deadRelus({ history, config: reluCfg })?.severity).toBe('critical')
  })
  it('ignores tanh layers and healthy ReLU layers', () => {
    const h1 = mkHistory(tanhCfg, 10, () => ({ layerOver: () => ({ deadPct: 0.9 }) }))
    expect(deadRelus({ history: h1, config: tanhCfg })).toBeNull()
    expect(deadRelus({ history: healthy(), config: reluCfg })).toBeNull()
  })
})

describe('learning rate too high', () => {
  it('fires on a zig-zagging loss and suggests a safe lr with a restart', () => {
    const history = mkHistory(reluCfg, 30, (i) => ({ trainLoss: i % 2 ? 1.4 : 0.6 }))
    const d = learningRateTooHigh({ history, config: reluCfg })!
    expect(d.evidence.chart).toBe('loss')
    expect(d.fix?.patch.lr).toBeCloseTo(0.01, 5)
    expect(d.fix?.reset).toBe(true)
  })
  it('caps the suggestion at a stable rate for the optimizer (adam at lr 1.5 -> 0.01)', () => {
    const cfg = { ...reluCfg, lr: 1.5, optimizer: 'adam' as const }
    const history = mkHistory(cfg, 30, (i) => ({ trainLoss: i % 2 ? 1.4 : 0.6 }))
    expect(learningRateTooHigh({ history, config: cfg })?.fix?.patch.lr).toBeCloseTo(0.01, 5)
    const sgd = { ...cfg, optimizer: 'sgd' as const }
    expect(learningRateTooHigh({ history, config: sgd })?.fix?.patch.lr).toBeCloseTo(0.1, 5)
  })
  it('fires when the loss climbs after having been low', () => {
    const history = mkHistory(reluCfg, 30, (i) => ({
      trainLoss: i < 15 ? 0.5 - i * 0.02 : 0.2 + (i - 15) * 0.1,
    }))
    expect(learningRateTooHigh({ history, config: reluCfg })).not.toBeNull()
  })
  it('does not fire on a smooth decrease', () => {
    expect(learningRateTooHigh({ history: healthy(), config: reluCfg })).toBeNull()
  })
  it('does not fire when there is too little history', () => {
    const h = mkHistory(reluCfg, 5, (i) => ({ trainLoss: i % 2 ? 1.4 : 0.6 }))
    expect(learningRateTooHigh({ history: h, config: reluCfg })).toBeNull()
  })
})

describe('learning rate too low', () => {
  it('fires when a high loss barely moves for a long time', () => {
    const history = mkHistory(reluCfg, 60, (i) => ({
      trainLoss: 0.69 - i * 0.00002,
      trainAcc: 0.5,
    }))
    const d = learningRateTooLow({ history, config: reluCfg })!
    expect(d.fix?.patch.lr).toBeCloseTo(0.09, 5)
    expect(d.fix?.reset).toBe(false)
  })
  it('does not fire when the loss is still falling', () => {
    expect(learningRateTooLow({ history: healthy(reluCfg, 60), config: reluCfg })).toBeNull()
  })
  it('does not fire when the loss is already low (converged)', () => {
    const h = mkHistory(reluCfg, 60, () => ({ trainLoss: 0.05, trainAcc: 0.99 }))
    expect(learningRateTooLow({ history: h, config: reluCfg })).toBeNull()
  })
  it('does not fire too early in training', () => {
    const h = mkHistory(reluCfg, 20, () => ({ trainLoss: 0.69, trainAcc: 0.5 }))
    expect(learningRateTooLow({ history: h, config: reluCfg })).toBeNull()
  })
})

describe('overfitting', () => {
  const overfit = mkHistory(reluCfg, 60, (i) => ({
    trainLoss: Math.max(0.02, 0.6 - i * 0.012),
    testLoss: i < 25 ? 0.6 - i * 0.01 : 0.35 + (i - 25) * 0.02,
    trainAcc: 0.99,
    testAcc: 0.8,
  }))
  it('fires when test loss rises while train loss falls, pointing at the best test step', () => {
    const d = overfitting({ history: overfit, config: reluCfg })!
    // test loss is 0.35 at index 25 (0.6 - 25*0.01 = 0.35 and 0.35 + 0*0.02 = 0.35): the minimum
    expect(d.evidence.step).toBe(overfit[25].step)
    expect(d.fix?.patch.l2).toBe(0.01)
    expect(d.fix?.reset).toBe(false)
  })
  it('suggests a smaller network when L2 is already on', () => {
    const cfg = { ...reluCfg, l2: 0.01 }
    const d = overfitting({ history: overfit, config: cfg })!
    expect(d.fix?.patch.layers?.[0].units).toBe(4)
    expect(d.fix?.reset).toBe(true)
  })
  it('does not fire for a healthy run', () => {
    expect(overfitting({ history: healthy(), config: reluCfg })).toBeNull()
  })
})

describe('saturated units', () => {
  it('fires when most tanh outputs are pinned at ±1', () => {
    const history = mkHistory(tanhCfg, 12, (i) => ({
      layerOver: (l) => (l === 0 ? { satPct: i < 4 ? 0.1 : 0.9 } : {}),
    }))
    const d = saturatedUnits({ history, config: tanhCfg })!
    expect(d.severity).toBe('critical')
    expect(d.evidence.step).toBe(history[4].step)
    expect(d.fix?.patch.layers?.every((l) => l.activation === 'relu')).toBe(true)
  })
  it('allows saturation once the task is learned (low loss)', () => {
    const history = mkHistory(tanhCfg, 12, () => ({
      trainLoss: 0.05,
      layerOver: () => ({ satPct: 0.9 }),
    }))
    expect(saturatedUnits({ history, config: tanhCfg })).toBeNull()
  })
  it('does not fire for ReLU nets or low saturation', () => {
    expect(saturatedUnits({ history: healthy(), config: reluCfg })).toBeNull()
    expect(saturatedUnits({ history: healthy(tanhCfg), config: tanhCfg })).toBeNull()
  })
})

describe('determinism', () => {
  it('same history gives the same diagnoses', () => {
    const history = mkHistory(sigmoidCfg, 30, () => ({
      trainLoss: 0.69,
      layerOver: (l) => ({ gNorm: [1e-6, 1e-5, 1e-4, 0.3][l], satPct: 0.9 }),
    }))
    const a = diagnose({ history, config: sigmoidCfg })
    const b = diagnose({ history, config: sigmoidCfg })
    expect(a).toEqual(b)
    expect(a.length).toBeGreaterThan(0)
  })
})
