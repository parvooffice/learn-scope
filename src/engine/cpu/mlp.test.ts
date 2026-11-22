import { describe, expect, it } from 'vitest'
import { CpuMLP } from './mlp'
import { generateDataset } from '../datasets/toy'

describe('CpuMLP', () => {
  it('analytic gradients match numerical gradients', () => {
    const net = new CpuMLP([2, 4, 3, 1], 5)
    const ds = generateDataset('moons', 20, 0.1, 2)
    const { dW, dB } = net.gradients(ds.x, ds.y)
    const h = 1e-6
    const check = (arr: Float64Array, grad: Float64Array) => {
      for (let i = 0; i < arr.length; i++) {
        const o = arr[i]
        arr[i] = o + h
        const lp = net.gradients(ds.x, ds.y).loss
        arr[i] = o - h
        const lm = net.gradients(ds.x, ds.y).loss
        arr[i] = o
        expect(Math.abs(grad[i] - (lp - lm) / (2 * h))).toBeLessThan(1e-6)
      }
    }
    net.W.forEach((w, l) => check(w, dW[l]))
    net.b.forEach((b, l) => check(b, dB[l]))
  })

  it('learns the spiral well enough for the hero animation', () => {
    const net = new CpuMLP([2, 10, 10, 1], 3)
    const ds = generateDataset('spiral', 160, 0.05, 1)
    let first = 0
    let last = 0
    for (let i = 0; i < 900; i++) {
      const l = net.trainStep(ds.x, ds.y, 0.03)
      if (i === 0) first = l
      last = l
    }
    expect(last).toBeLessThan(first * 0.5)
  })
})
