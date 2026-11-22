import { describe, expect, it } from 'vitest'
import { Value } from './value'
import { MLP, trainMSE } from './nn'

/** Central-difference numerical gradient of f at point xs, for each coordinate. */
function numericalGrad(f: (xs: Value[]) => Value, xs: number[], h = 1e-6): number[] {
  return xs.map((_, i) => {
    const plus = xs.slice()
    const minus = xs.slice()
    plus[i] += h
    minus[i] -= h
    const fp = f(plus.map((x) => new Value(x))).data
    const fm = f(minus.map((x) => new Value(x))).data
    return (fp - fm) / (2 * h)
  })
}

function analyticGrad(f: (xs: Value[]) => Value, xs: number[]): number[] {
  const inputs = xs.map((x) => new Value(x))
  f(inputs).backward()
  return inputs.map((i) => i.grad)
}

function expectGradsMatch(f: (xs: Value[]) => Value, xs: number[]) {
  const a = analyticGrad(f, xs)
  const n = numericalGrad(f, xs)
  a.forEach((g, i) => expect(Math.abs(g - n[i])).toBeLessThan(1e-6))
}

describe('Value autograd: gradients match numerical differentiation (<1e-6)', () => {
  it('add / mul', () => expectGradsMatch(([a, b, c]) => a.mul(b).add(c).mul(a), [0.3, -1.2, 2.5]))
  it('sub / div', () => expectGradsMatch(([a, b]) => a.sub(b).div(b.mul(b).add(1)), [1.7, -0.4]))
  it('pow', () => expectGradsMatch(([a, b]) => a.pow(3).add(b.pow(2)), [0.8, -1.1]))
  it('tanh', () => expectGradsMatch(([a, b]) => a.mul(b).tanh(), [0.5, -0.9]))
  it('sigmoid', () => expectGradsMatch(([a, b]) => a.add(b).sigmoid(), [0.5, -0.9]))
  it('exp / log', () => expectGradsMatch(([a, b]) => a.exp().add(b.mul(b).add(1).log()), [0.2, 1.3]))
  it('relu (away from the kink)', () =>
    expectGradsMatch(([a, b]) => a.mul(b).relu().add(b.relu()), [0.7, 1.4]))
  it('a value used twice accumulates gradients', () => {
    const a = new Value(3)
    const out = a.mul(a) // a^2 -> d/da = 2a = 6
    out.backward()
    expect(a.grad).toBeCloseTo(6, 10)
  })
  it('a whole MLP loss matches numerical gradients', () => {
    const net = new MLP([2, 3, 1], ['tanh', 'tanh'], 7)
    const xs = [
      [0.5, -0.3],
      [-1, 0.8],
    ]
    const ys = [1, -1]
    const loss = () => {
      let l = new Value(0)
      xs.forEach((x, i) => (l = l.add(net.call(x)[0].sub(ys[i]).pow(2))))
      return l
    }
    net.zeroGrad()
    loss().backward()
    const params = net.params()
    const analytic = params.map((p) => p.grad)
    const h = 1e-6
    params.forEach((p, i) => {
      const orig = p.data
      p.data = orig + h
      const fp = loss().data
      p.data = orig - h
      const fm = loss().data
      p.data = orig
      expect(Math.abs(analytic[i] - (fp - fm) / (2 * h))).toBeLessThan(1e-6)
    })
  })
})

describe('Tiny MLP', () => {
  it('learns XOR', () => {
    const net = new MLP([2, 4, 1], ['tanh', 'tanh'], 3)
    const xs = [
      [-1, -1],
      [-1, 1],
      [1, -1],
      [1, 1],
    ]
    const ys = [-1, 1, 1, -1]
    const hist = trainMSE(net, xs, ys, { lr: 0.1, steps: 400 })
    expect(hist[hist.length - 1]).toBeLessThan(0.02)
    xs.forEach((x, i) => expect(Math.sign(net.call(x)[0].data)).toBe(ys[i]))
  })
})
