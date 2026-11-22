import { useEffect, useMemo, useRef, useState } from 'react'
import {
  applyStep,
  initialParams,
  runExplainer,
  type ExplainerParams,
} from '../../engine/tiny/explainer'
import { Btn, Panel, Slider } from '../../components/ui'
import { LineChart } from '../../viz/LineChart'

const f = (n: number, d = 4) => (Object.is(n, -0) ? 0 : n).toFixed(d)
const sub = (n: number) => '₀₁₂₃₄₅₆₇₈₉'[n]
const sgn = (n: number) => (n < 0 ? `(${f(n)})` : f(n))

interface StepInfo {
  phase: 'setup' | 'forward' | 'loss' | 'backward' | 'update'
  title: string
  idea: string
  lines: string[]
}

const NODE = {
  x1: { x: 70, y: 105 },
  x2: { x: 70, y: 255 },
  h: [
    { x: 290, y: 70 },
    { x: 290, y: 180 },
    { x: 290, y: 295 },
  ],
  y: { x: 500, y: 180 },
  L: { x: 640, y: 180 },
}

const STEP_COUNT = 10

export default function Backprop() {
  const [params, setParams] = useState<ExplainerParams>(() => initialParams(4))
  const [lr, setLr] = useState(0.2)
  const [step, setStep] = useState(0)
  const [auto, setAuto] = useState(false)
  const [losses, setLosses] = useState<number[]>([])
  const r = useMemo(() => runExplainer(params), [params])
  const next = useMemo(() => applyStep(params, r.auto, lr), [params, r, lr])
  const rNext = useMemo(() => runExplainer(next), [next])
  const timer = useRef<ReturnType<typeof setInterval> | undefined>(undefined)

  useEffect(() => {
    if (!auto) return
    timer.current = setInterval(
      () => setStep((s) => (s >= STEP_COUNT - 1 ? (setAuto(false), s) : s + 1)),
      1700,
    )
    return () => clearInterval(timer.current)
  }, [auto])

  const { x, t, w, b, v, c } = params
  const steps: StepInfo[] = [
    {
      phase: 'setup',
      title: '0 · The network and its numbers',
      idea: 'A neural network is just a formula with adjustable numbers (weights w, biases b, v, c). Training means nudging those numbers so the output ŷ gets close to the target t. Below, everything is computed live by our own autograd engine.',
      lines: [
        `inputs: x₁ = ${f(x[0], 2)}, x₂ = ${f(x[1], 2)}      target: t = ${f(t, 2)}`,
        `hidden layer (3 neurons, tanh):  z = w·x + b,   h = tanh(z)`,
        `output (linear):  ŷ = v₁h₁ + v₂h₂ + v₃h₃ + c`,
        `loss:  L = (ŷ − t)²   (how wrong we are; we want it near 0)`,
      ],
    },
    {
      phase: 'forward',
      title: '1 · Forward pass: weighted sums',
      idea: 'Each hidden neuron multiplies every input by a weight, adds them up, and adds its bias. This is the weighted sum z.',
      lines: r.z.map(
        (z, j) =>
          `z${sub(j + 1)} = w${sub(j + 1)}₁·x₁ + w${sub(j + 1)}₂·x₂ + b${sub(j + 1)} = ${sgn(w[j][0])}(${f(x[0], 2)}) + ${sgn(w[j][1])}(${f(x[1], 2)}) + ${sgn(b[j])} = ${f(z)}`,
      ),
    },
    {
      phase: 'forward',
      title: '2 · Forward pass: the activation',
      idea: 'tanh squashes z into the range −1…1. Without a bend like this, stacking layers would still be just one straight line.',
      lines: r.h.map((h, j) => `h${sub(j + 1)} = tanh(${f(r.z[j])}) = ${f(h)}`),
    },
    {
      phase: 'forward',
      title: '3 · Forward pass: the output',
      idea: 'The output neuron takes a weighted sum of the hidden values. That is our prediction ŷ.',
      lines: [
        `ŷ = ${v.map((vj, j) => `${sgn(vj)}(${f(r.h[j])})`).join(' + ')} + ${sgn(c)}`,
        `ŷ = ${f(r.yhat)}`,
      ],
    },
    {
      phase: 'loss',
      title: '4 · The loss',
      idea: 'We measure the mistake with the squared error. Squaring makes it always positive and punishes big mistakes more. Learning = making this number smaller.',
      lines: [`L = (ŷ − t)² = (${f(r.yhat)} − ${f(t, 2)})² = ${f(r.loss)}`],
    },
    {
      phase: 'backward',
      title: '5 · Backward pass: start at the loss',
      idea: 'Backpropagation asks: "if I nudge ŷ a tiny bit, how much does L change?" That sensitivity is the derivative ∂L/∂ŷ. It is the first link of the chain; every other gradient is built from it.',
      lines: [
        `∂L/∂ŷ = 2(ŷ − t) = 2(${f(r.yhat)} − ${f(t, 2)}) = ${f(r.manual.yhat)}`,
        `autograd says: ${f(r.auto.yhat)}`,
      ],
    },
    {
      phase: 'backward',
      title: '6 · Backward: through the output neuron',
      idea: 'Chain rule: (how L changes with ŷ) × (how ŷ changes with the thing). Since ŷ = Σ v·h + c, the "how ŷ changes with v" is just h, and with h it is just v.',
      lines: [
        ...r.h.map(
          (h, j) =>
            `∂L/∂v${sub(j + 1)} = ∂L/∂ŷ · h${sub(j + 1)} = ${f(r.manual.yhat)} · ${f(h)} = ${f(r.manual.v[j])}`,
        ),
        `∂L/∂c = ∂L/∂ŷ · 1 = ${f(r.manual.c)}`,
        ...v.map(
          (vj, j) =>
            `∂L/∂h${sub(j + 1)} = ∂L/∂ŷ · v${sub(j + 1)} = ${f(r.manual.yhat)} · ${sgn(vj)} = ${f(r.manual.h[j])}`,
        ),
      ],
    },
    {
      phase: 'backward',
      title: '7 · Backward: through tanh',
      idea: 'The slope of tanh at z is 1 − tanh(z)² = 1 − h². Where tanh is flat (h near ±1) the slope is near 0 and the gradient gets squashed. That is why deep tanh/sigmoid networks suffer vanishing gradients.',
      lines: r.h.map(
        (h, j) =>
          `∂L/∂z${sub(j + 1)} = ∂L/∂h${sub(j + 1)} · (1 − h${sub(j + 1)}²) = ${f(r.manual.h[j])} · ${f(1 - h * h)} = ${f(r.manual.z[j])}`,
      ),
    },
    {
      phase: 'backward',
      title: '8 · Backward: to the first-layer weights',
      idea: 'Since z = w₁x₁ + w₂x₂ + b, the slope of z with respect to w₁ is x₁, and with respect to b is 1. Now every weight in the network knows which way (and how strongly) it should move.',
      lines: [
        ...r.h.flatMap((_, j) => [
          `∂L/∂w${sub(j + 1)}₁ = ${f(r.manual.z[j])} · ${f(x[0], 2)} = ${f(r.manual.w[j][0])}      ∂L/∂w${sub(j + 1)}₂ = ${f(r.manual.z[j])} · ${f(x[1], 2)} = ${f(r.manual.w[j][1])}      ∂L/∂b${sub(j + 1)} = ${f(r.manual.b[j])}`,
        ]),
        `✓ hand-written chain rule vs autograd engine: max difference = ${r.maxDiff.toExponential(1)}`,
      ],
    },
    {
      phase: 'update',
      title: '9 · The update: gradient descent',
      idea: `Move every parameter a small step against its gradient: p ← p − lr · ∂L/∂p (lr = ${lr}). Then the loss goes down. Repeat thousands of times: that is training.`,
      lines: [
        `w₁₁: ${f(w[0][0])} → ${f(next.w[0][0])}     v₁: ${f(v[0])} → ${f(next.v[0])}     b₁: ${f(b[0])} → ${f(next.b[0])}   (… all ${9 + 3 + 3 + 1} parameters)`,
        `loss before = ${f(r.loss, 5)}   →   loss after = ${f(rNext.loss, 5)}   ${rNext.loss < r.loss ? '✓ lower' : '⚠ higher: lr too big'}`,
      ],
    },
  ]
  const cur = steps[step]

  // visibility
  const showZ = step >= 1
  const showH = step >= 2
  const showY = step >= 3
  const showL = step >= 4
  const g1 = step >= 5 // dL/dyhat
  const g2 = step >= 6 // dL/dv, dL/dh
  const g3 = step >= 7 // dL/dz
  const g4 = step >= 8 // dL/dw
  const active = (s: number) => step === s

  const applyUpdate = () => {
    setLosses((l) => [...l, r.loss])
    setParams(next)
    setStep(0)
  }
  const train = (n: number) => {
    let p = params
    const hist: number[] = []
    for (let i = 0; i < n; i++) {
      const res = runExplainer(p)
      hist.push(res.loss)
      p = applyStep(p, res.auto, lr)
    }
    setLosses((l) => [...l, ...hist])
    setParams(p)
    setStep(0)
  }
  const reset = () => {
    setParams(initialParams(4))
    setLosses([])
    setStep(0)
    setAuto(false)
  }
  const setX = (i: 0 | 1, val: number) =>
    setParams((p) => ({ ...p, x: i === 0 ? [val, p.x[1]] : [p.x[0], val] }))

  const edge = (
    x1: number,
    y1: number,
    x2: number,
    y2: number,
    wv: number,
    hi: boolean,
    grad?: number,
    tpos = 0.5,
  ) => {
    const mx = x1 + (x2 - x1) * tpos
    const my = y1 + (y2 - y1) * tpos
    const sw = 0.8 + Math.min(4, Math.abs(wv) * 3)
    return (
      <g key={`${x1}-${y1}-${x2}-${y2}`}>
        <line
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke={wv >= 0 ? 'var(--pos)' : 'var(--neg)'}
          strokeWidth={hi ? sw + 2 : sw}
          opacity={hi ? 1 : 0.5}
        />
        <text
          x={mx}
          y={my - 3}
          textAnchor="middle"
          fontSize="10"
          className="num"
          fill="var(--text)"
          stroke="var(--panel)"
          strokeWidth="3"
          paintOrder="stroke"
        >
          {f(wv, 2)}
        </text>
        {grad !== undefined && (
          <text
            x={mx}
            y={my + 10}
            textAnchor="middle"
            fontSize="10"
            className="num"
            fill="var(--warn)"
            stroke="var(--panel)"
            strokeWidth="3"
            paintOrder="stroke"
          >
            ∂ {f(grad, 3)}
          </text>
        )}
      </g>
    )
  }

  const node = (
    id: string,
    cx: number,
    cy: number,
    top: string,
    lines: { t: string; color?: string }[],
    hi: boolean,
    rad = 24,
  ) => (
    <g key={id}>
      <circle
        cx={cx}
        cy={cy}
        r={rad}
        fill="var(--panel2)"
        stroke={hi ? 'var(--accent)' : 'var(--border)'}
        strokeWidth={hi ? 3 : 1.5}
      />
      <text x={cx} y={cy + 4} textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--text)">
        {top}
      </text>
      {lines.map((ln, i) => (
        <text
          key={i}
          x={cx}
          y={cy + rad + 14 + i * 12}
          textAnchor="middle"
          fontSize="10"
          className="num"
          fill={ln.color ?? 'var(--muted)'}
        >
          {ln.t}
        </text>
      ))}
    </g>
  )

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[1280px] space-y-3 p-3">
        <div>
          <h1 className="text-xl font-semibold">Backpropagation, step by step</h1>
          <p className="text-sm text-muted">
            A 2-3-1 network with real numbers. Press <b>Next</b> to walk through one full training
            step: forward, loss, backward, update.
            <span className="text-accent"> Blue/red lines are weights;</span>{' '}
            <span className="text-warn">amber ∂ values are gradients.</span>
          </p>
        </div>

        <div className="grid gap-3 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
          <Panel
            title="The network"
            info="Each circle is a neuron and each line a weight (red = positive, blue = negative). Values appear as the computation reaches them. During the backward pass, amber numbers show the gradient ∂L/∂(thing): how much the loss would change if that thing changed a little."
          >
            <svg
              viewBox="0 0 710 410"
              className="w-full"
              role="img"
              aria-label="Diagram of the 2-3-1 network showing values and gradients for the current step"
            >
              {/* input -> hidden edges */}
              {NODE.h.map((h, j) => [
                edge(
                  NODE.x1.x + 24,
                  NODE.x1.y,
                  h.x - 24,
                  h.y,
                  w[j][0],
                  active(1) || active(8),
                  g4 ? r.manual.w[j][0] : undefined,
                  0.3,
                ),
                edge(
                  NODE.x2.x + 24,
                  NODE.x2.y,
                  h.x - 24,
                  h.y,
                  w[j][1],
                  active(1) || active(8),
                  g4 ? r.manual.w[j][1] : undefined,
                  0.3,
                ),
              ])}
              {/* hidden -> output edges */}
              {NODE.h.map((h, j) =>
                edge(
                  h.x + 24,
                  h.y,
                  NODE.y.x - 24,
                  NODE.y.y,
                  v[j],
                  active(3) || active(6),
                  g2 ? r.manual.v[j] : undefined,
                  0.5,
                ),
              )}
              <line
                x1={NODE.y.x + 24}
                y1={NODE.y.y}
                x2={NODE.L.x - 24}
                y2={NODE.L.y}
                stroke="var(--muted)"
                strokeWidth={active(4) || active(5) ? 4 : 1.5}
                opacity={0.7}
              />

              {node(
                'x1',
                NODE.x1.x,
                NODE.x1.y,
                'x₁',
                [{ t: f(x[0], 2), color: 'var(--text)' }],
                false,
              )}
              {node(
                'x2',
                NODE.x2.x,
                NODE.x2.y,
                'x₂',
                [{ t: f(x[1], 2), color: 'var(--text)' }],
                false,
              )}
              {NODE.h.map((h, j) =>
                node(
                  `h${j}`,
                  h.x,
                  h.y,
                  `h${sub(j + 1)}`,
                  [
                    ...(showZ ? [{ t: `z = ${f(r.z[j], 3)}` }] : [{ t: `b = ${f(b[j], 2)}` }]),
                    ...(showH ? [{ t: `h = ${f(r.h[j], 3)}`, color: 'var(--text)' }] : []),
                    ...(g2 && !g3
                      ? [{ t: `∂h ${f(r.manual.h[j], 3)}`, color: 'var(--warn)' }]
                      : []),
                    ...(g3 ? [{ t: `∂z ${f(r.manual.z[j], 3)}`, color: 'var(--warn)' }] : []),
                    ...(g4 ? [{ t: `∂b ${f(r.manual.b[j], 3)}`, color: 'var(--warn)' }] : []),
                  ],
                  active(2) || active(7),
                ),
              )}
              {node(
                'y',
                NODE.y.x,
                NODE.y.y,
                'ŷ',
                [
                  ...(showY
                    ? [{ t: `ŷ = ${f(r.yhat, 3)}`, color: 'var(--text)' }]
                    : [{ t: `c = ${f(c, 2)}` }]),
                  ...(g1 ? [{ t: `∂ŷ ${f(r.manual.yhat, 3)}`, color: 'var(--warn)' }] : []),
                  ...(g2 ? [{ t: `∂c ${f(r.manual.c, 3)}`, color: 'var(--warn)' }] : []),
                ],
                active(3) || active(5) || active(6),
              )}
              {node(
                'L',
                NODE.L.x,
                NODE.L.y,
                'L',
                [
                  { t: `t = ${f(t, 2)}` },
                  ...(showL ? [{ t: `L = ${f(r.loss, 4)}`, color: 'var(--text)' }] : []),
                ],
                active(4),
                22,
              )}
              <text x="355" y="16" textAnchor="middle" fontSize="11" fill="var(--muted)">
                {cur.phase === 'backward'
                  ? '⟵ gradients flow backward'
                  : cur.phase === 'forward'
                    ? 'values flow forward ⟶'
                    : ''}
              </text>
            </svg>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Btn
                onClick={() => setStep((s) => Math.max(0, s - 1))}
                disabled={step === 0}
                ariaLabel="Previous step"
              >
                ← Back
              </Btn>
              <Btn
                variant="primary"
                onClick={() => setStep((s) => Math.min(STEP_COUNT - 1, s + 1))}
                disabled={step === STEP_COUNT - 1}
                ariaLabel="Next step"
              >
                Next →
              </Btn>
              <Btn onClick={() => setAuto((a) => !a)}>{auto ? '⏸ Pause' : '▶ Auto-play'}</Btn>
              <span className="num text-xs text-muted">
                step {step + 1} / {STEP_COUNT}
              </span>
              <div className="ml-auto flex gap-2">
                <Btn onClick={applyUpdate} title="Apply the update and start the next iteration">
                  Apply update ⟳
                </Btn>
                <Btn onClick={() => train(100)} title="Run 100 full training steps instantly">
                  Train ×100
                </Btn>
                <Btn variant="danger" onClick={reset}>
                  Reset
                </Btn>
              </div>
            </div>
          </Panel>

          <div className="space-y-3">
            <Panel
              title={cur.title}
              info="This panel explains the current step and shows the exact arithmetic with the live numbers from the diagram."
            >
              <p className="text-sm leading-relaxed">{cur.idea}</p>
              <pre
                className="num scroll-thin mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-panel2 p-2 text-[11.5px] leading-relaxed text-ink"
                aria-live="polite"
              >
                {cur.lines.join('\n')}
              </pre>
            </Panel>
            <Panel
              title="Play with it"
              info="Change the inputs, the target or the learning rate; every number above is recomputed instantly by the autograd engine."
            >
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <Slider
                  label="x₁"
                  value={x[0]}
                  min={-1.5}
                  max={1.5}
                  step={0.05}
                  onChange={(val) => setX(0, val)}
                  format={(n) => n.toFixed(2)}
                />
                <Slider
                  label="x₂"
                  value={x[1]}
                  min={-1.5}
                  max={1.5}
                  step={0.05}
                  onChange={(val) => setX(1, val)}
                  format={(n) => n.toFixed(2)}
                />
                <Slider
                  label="target t"
                  value={t}
                  min={-1.5}
                  max={1.5}
                  step={0.05}
                  onChange={(val) => setParams((p) => ({ ...p, t: val }))}
                  format={(n) => n.toFixed(2)}
                />
                <Slider
                  label="learning rate"
                  value={lr}
                  min={0.01}
                  max={1.5}
                  step={0.01}
                  onChange={setLr}
                  format={(n) => n.toFixed(2)}
                />
              </div>
            </Panel>
            <Panel
              title="Loss over updates"
              info="Each time you apply an update the loss is recorded. A well-chosen learning rate makes it fall smoothly toward zero; a too-large learning rate makes it bounce or blow up (try 1.5)."
            >
              {losses.length > 1 ? (
                <LineChart
                  series={[
                    { label: 'loss', color: '#2dd4bf', x: losses.map((_, i) => i), y: losses },
                  ]}
                  label="Loss after each update"
                  height={120}
                  yMin={0}
                />
              ) : (
                <p className="text-xs text-muted">
                  Press “Apply update” or “Train ×100” to see the loss fall.
                </p>
              )}
            </Panel>
          </div>
        </div>
      </div>
    </div>
  )
}
