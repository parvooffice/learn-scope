import { describe, expect, it } from 'vitest'
import * as tf from '@tensorflow/tfjs'
import { MnistTrainer, type MnistDataset } from './mnistTrainer'
import { mulberry32 } from '../../lib/rng'

/** Synthetic "digits": class c lights up a 6×6 block at a class-specific position, plus noise. */
function synth(nTrain: number, nTest: number): MnistDataset {
  const rng = mulberry32(9)
  const n = nTrain + nTest
  const images = new Uint8Array(n * 784)
  const labels = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    const c = i % 10
    labels[i] = c
    const bx = 2 + (c % 5) * 5
    const by = 4 + Math.floor(c / 5) * 12
    for (let y = 0; y < 28; y++)
      for (let x = 0; x < 28; x++) {
        const inside = x >= bx && x < bx + 6 && y >= by && y < by + 6
        images[i * 784 + y * 28 + x] = Math.min(255, (inside ? 230 : 0) + rng() * 25)
      }
  }
  return { images, labels, nTrain, nTest }
}

describe('MnistTrainer (CPU backend, synthetic data)', () => {
  for (const model of ['cnn', 'mlp'] as const) {
    it(`${model}: trains, evaluates and exposes visualisation data`, async () => {
      await tf.setBackend('cpu')
      await tf.ready()
      const data = synth(200, 100)
      const t = new MnistTrainer(tf, { model, lr: 0.01, batchSize: 20, seed: 1 }, data)
      const first = await t.snapshot(true)
      for (let i = 0; i < (model === 'cnn' ? 25 : 40); i++) await t.trainStep()
      const snap = await t.snapshot(true)
      expect(snap.step).toBeGreaterThan(0)
      expect(snap.eval!.testAcc).toBeGreaterThan(first.eval!.testAcc)
      expect(snap.eval!.confusion).toHaveLength(100)
      expect(snap.eval!.confusion.reduce((a, b) => a + b, 0)).toBe(100)
      expect(snap.filters.count).toBe(model === 'cnn' ? 8 : 16)
      expect(snap.filters.data.length).toBe(snap.filters.count * snap.filters.size ** 2)
      const focus = await t.focus(new Float32Array(784).fill(0.1))
      expect(focus.probs).toHaveLength(10)
      expect(focus.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4)
      if (model === 'cnn') {
        expect(focus.conv1).toMatchObject({ count: 8, size: 24 })
        expect(focus.conv2).toMatchObject({ count: 16, size: 10 })
      } else expect(focus.hidden).toHaveLength(128)
      t.dispose()
    }, 60000)
  }
})
