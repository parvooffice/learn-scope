import { describe, expect, it } from 'vitest'
import * as tf from '@tensorflow/tfjs'
import { Trainer } from './trainer'
import { buildLayerStats } from './layerStats'
import { DEFAULT_CONFIG, type PlaygroundConfig } from '../types'
import { generateDataset, splitDataset } from '../datasets/toy'
import { evaluate } from '../forward'

async function setup(over: Partial<PlaygroundConfig>) {
  await tf.setBackend('cpu')
  await tf.ready()
  const cfg = { ...DEFAULT_CONFIG, ...over }
  const ds = generateDataset(cfg.dataset, cfg.samples, cfg.noise, cfg.seed)
  const { train, test } = splitDataset(ds, cfg.testSplit, cfg.seed)
  return { cfg, train, test, trainer: new Trainer(tf, cfg, train, test) }
}

describe('Trainer (TF.js, CPU backend in tests)', () => {
  it('learns two blobs quickly with Adam', async () => {
    const { trainer } = await setup({
      dataset: 'blobs',
      layers: [{ units: 4, activation: 'relu' }],
      lr: 0.05,
      samples: 200,
    })
    const first = await trainer.snapshot()
    for (let i = 0; i < 80; i++) trainer.trainStep()
    const last = await trainer.snapshot()
    expect(last.step).toBe(80)
    expect(last.trainLoss).toBeLessThan(first.trainLoss)
    expect(last.testAcc).toBeGreaterThan(0.9)
    trainer.dispose()
  })

  for (const optimizer of ['sgd', 'momentum', 'adam'] as const) {
    it(`${optimizer} reduces the loss`, async () => {
      const { trainer } = await setup({
        dataset: 'moons',
        optimizer,
        lr: optimizer === 'sgd' ? 0.3 : 0.03,
        layers: [{ units: 6, activation: 'tanh' }],
      })
      const first = await trainer.snapshot()
      for (let i = 0; i < 120; i++) trainer.trainStep()
      const last = await trainer.snapshot()
      expect(last.trainLoss).toBeLessThan(first.trainLoss)
      trainer.dispose()
    })
  }

  it('TF metrics agree with the independent CPU forward pass (all 4 activations)', async () => {
    const { trainer, cfg, train, test } = await setup({
      layers: [
        { units: 5, activation: 'relu' },
        { units: 4, activation: 'gelu' },
        { units: 3, activation: 'tanh' },
        { units: 3, activation: 'sigmoid' },
      ],
    })
    for (let i = 0; i < 10; i++) trainer.trainStep()
    const snap = await trainer.snapshot()
    const trRef = evaluate(trainer.specs, snap.weights, train.x, train.y)
    const teRef = evaluate(trainer.specs, snap.weights, test.x, test.y)
    expect(snap.trainLoss).toBeCloseTo(trRef.loss, 3)
    expect(snap.testLoss).toBeCloseTo(teRef.loss, 3)
    expect(snap.trainAcc).toBeCloseTo(trRef.acc, 5)
    expect(snap.layers).toHaveLength(cfg.layers.length + 1)
    trainer.dispose()
  })

  it('is deterministic for a fixed seed', async () => {
    const a = await setup({ dataset: 'xor' })
    const b = await setup({ dataset: 'xor' })
    for (let i = 0; i < 15; i++) {
      a.trainer.trainStep()
      b.trainer.trainStep()
    }
    const sa = await a.trainer.snapshot()
    const sb = await b.trainer.snapshot()
    expect(sa.trainLoss).toBeCloseTo(sb.trainLoss, 6)
    a.trainer.dispose()
    b.trainer.dispose()
  })

  it('full-batch gradient norms in the snapshot are finite and non-zero at init', async () => {
    const { trainer } = await setup({})
    const snap = await trainer.snapshot()
    for (const l of snap.layers) {
      expect(Number.isFinite(l.gNorm)).toBe(true)
      expect(l.gNorm).toBeGreaterThan(0)
    }
    trainer.dispose()
  })

  it('does not leak tensors across many steps', async () => {
    const { trainer } = await setup({ layers: [{ units: 4, activation: 'relu' }] })
    for (let i = 0; i < 5; i++) trainer.trainStep()
    await trainer.snapshot()
    const before = tf.memory().numTensors
    for (let i = 0; i < 30; i++) trainer.trainStep()
    await trainer.snapshot()
    const after = tf.memory().numTensors
    expect(after - before).toBeLessThanOrEqual(2)
    trainer.dispose()
  })
})

describe('buildLayerStats', () => {
  it('detects dead relu units and saturation', () => {
    // 2 units, 4 samples. unit 0 is always 0 (dead); unit 1 fires.
    const acts = [0, 1, 0, 2, 0, 0.5, 0, 3]
    const s = buildLayerStats({
      activation: 'relu',
      nout: 2,
      W: [0.1, -0.2, 0.3, 0.4],
      gradW: [0.01, 0.02, -0.01, 0],
      acts,
      n: 4,
    })
    expect(s.deadPct).toBeCloseTo(0.5)
    expect(s.zeroPct).toBeCloseTo(0.5)
    expect(s.unitMean[1]).toBeCloseTo(1.625)
    const t = buildLayerStats({
      activation: 'tanh',
      nout: 1,
      W: [1],
      gradW: [0],
      acts: [0.99, -0.99, 0.2, 0.97],
      n: 4,
    })
    expect(t.satPct).toBeCloseTo(0.75)
    expect(t.deadPct).toBe(0)
  })
})
