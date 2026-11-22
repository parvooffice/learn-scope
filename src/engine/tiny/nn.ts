import { Value } from './value'
import { mulberry32 } from '../../lib/rng'

export type TinyAct = 'tanh' | 'relu' | 'sigmoid' | 'linear'

function activate(z: Value, act: TinyAct): Value {
  switch (act) {
    case 'tanh':
      return z.tanh()
    case 'relu':
      return z.relu()
    case 'sigmoid':
      return z.sigmoid()
    default:
      return z
  }
}

export class Neuron {
  w: Value[]
  b: Value
  constructor(
    nin: number,
    readonly act: TinyAct,
    rng: () => number,
  ) {
    this.w = Array.from({ length: nin }, () => new Value(rng() * 2 - 1))
    this.b = new Value(0)
  }
  /** a = act( w·x + b ) */
  call(x: Value[]): Value {
    let z = this.b
    for (let i = 0; i < this.w.length; i++) z = z.add(this.w[i].mul(x[i]))
    return activate(z, this.act)
  }
  params(): Value[] {
    return [...this.w, this.b]
  }
}

export class Layer {
  neurons: Neuron[]
  constructor(nin: number, nout: number, act: TinyAct, rng: () => number) {
    this.neurons = Array.from({ length: nout }, () => new Neuron(nin, act, rng))
  }
  call(x: Value[]): Value[] {
    return this.neurons.map((n) => n.call(x))
  }
  params(): Value[] {
    return this.neurons.flatMap((n) => n.params())
  }
}

export class MLP {
  layers: Layer[]
  constructor(sizes: number[], acts: TinyAct[], seed = 1) {
    const rng = mulberry32(seed)
    this.layers = sizes.slice(1).map((n, i) => new Layer(sizes[i], n, acts[i], rng))
  }
  call(x: (number | Value)[]): Value[] {
    let a: Value[] = x.map((e) => Value.of(e))
    for (const l of this.layers) a = l.call(a)
    return a
  }
  params(): Value[] {
    return this.layers.flatMap((l) => l.params())
  }
  zeroGrad() {
    for (const p of this.params()) p.grad = 0
  }
}

/** Train on (x, y) pairs with mean squared error + plain SGD. Returns the loss history. */
export function trainMSE(
  net: MLP,
  xs: number[][],
  ys: number[],
  opts: { lr: number; steps: number },
): number[] {
  const history: number[] = []
  for (let s = 0; s < opts.steps; s++) {
    let loss = new Value(0)
    for (let i = 0; i < xs.length; i++) {
      const out = net.call(xs[i])[0]
      loss = loss.add(out.sub(ys[i]).pow(2))
    }
    loss = loss.mul(1 / xs.length)
    net.zeroGrad()
    loss.backward()
    for (const p of net.params()) p.data -= opts.lr * p.grad
    history.push(loss.data)
  }
  return history
}
