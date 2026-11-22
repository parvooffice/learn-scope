/**
 * A scalar that remembers how it was computed, so we can run backpropagation.
 *
 * Every operation (add, mul, tanh ...) creates a NEW Value that stores
 *   - its parents (`prev`)
 *   - a small closure `_backward` that knows the LOCAL derivative of that operation
 * Calling `backward()` on the final loss applies the chain rule through the whole graph.
 *
 * The idea (and some naming) follows Andrej Karpathy's "micrograd".
 */
export class Value {
  data: number
  grad = 0
  readonly op: string
  readonly prev: Value[]
  label: string
  _backward: () => void = () => {}

  constructor(data: number, prev: Value[] = [], op = '', label = '') {
    this.data = data
    this.prev = prev
    this.op = op
    this.label = label
  }

  static of(x: number | Value): Value {
    return x instanceof Value ? x : new Value(x)
  }

  /** out = a + b.   d(out)/da = 1,  d(out)/db = 1 */
  add(other: number | Value): Value {
    const o = Value.of(other)
    const out = new Value(this.data + o.data, [this, o], '+')
    out._backward = () => {
      this.grad += out.grad
      o.grad += out.grad
    }
    return out
  }

  /** out = a * b.   d(out)/da = b,  d(out)/db = a */
  mul(other: number | Value): Value {
    const o = Value.of(other)
    const out = new Value(this.data * o.data, [this, o], '×')
    out._backward = () => {
      this.grad += o.data * out.grad
      o.grad += this.data * out.grad
    }
    return out
  }

  /** out = a ^ k for a constant number k.   d(out)/da = k * a^(k-1) */
  pow(k: number): Value {
    const out = new Value(Math.pow(this.data, k), [this], `^${k}`)
    out._backward = () => {
      this.grad += k * Math.pow(this.data, k - 1) * out.grad
    }
    return out
  }

  /** out = tanh(a).   d(out)/da = 1 - tanh(a)^2 = 1 - out^2 */
  tanh(): Value {
    const t = Math.tanh(this.data)
    const out = new Value(t, [this], 'tanh')
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad
    }
    return out
  }

  /** out = max(0, a).   d(out)/da = 1 if a > 0 else 0 */
  relu(): Value {
    const out = new Value(this.data > 0 ? this.data : 0, [this], 'relu')
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad
    }
    return out
  }

  /** out = 1 / (1 + e^-a).   d(out)/da = out * (1 - out) */
  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data))
    const out = new Value(s, [this], 'sigmoid')
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad
    }
    return out
  }

  /** out = e^a.   d(out)/da = e^a = out */
  exp(): Value {
    const e = Math.exp(this.data)
    const out = new Value(e, [this], 'exp')
    out._backward = () => {
      this.grad += e * out.grad
    }
    return out
  }

  /** out = ln(a).   d(out)/da = 1 / a */
  log(): Value {
    const out = new Value(Math.log(this.data), [this], 'log')
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad
    }
    return out
  }

  neg(): Value {
    return this.mul(-1)
  }
  sub(other: number | Value): Value {
    return this.add(Value.of(other).neg())
  }
  div(other: number | Value): Value {
    return this.mul(Value.of(other).pow(-1))
  }

  /** Nodes in dependency order: every node appears AFTER all of its parents. */
  topo(): Value[] {
    const order: Value[] = []
    const seen = new Set<Value>()
    const visit = (v: Value) => {
      if (seen.has(v)) return
      seen.add(v)
      for (const p of v.prev) visit(p)
      order.push(v)
    }
    visit(this)
    return order
  }

  /** Run backpropagation: set d(this)/d(this) = 1, then apply the chain rule in reverse order. */
  backward(): void {
    const order = this.topo()
    for (const v of order) v.grad = 0
    this.grad = 1
    for (let i = order.length - 1; i >= 0; i--) order[i]._backward()
  }
}

export const v = (x: number, label = '') => new Value(x, [], '', label)
