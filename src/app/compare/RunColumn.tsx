import { useMemo } from 'react'
import type { RunHandle } from './runs'
import { Badge, Btn, Panel, Select, Slider } from '../../components/ui'
import { DecisionBoundary, EXTENT, GRID } from '../../viz/DecisionBoundary'
import { NetworkGraph } from '../../viz/NetworkGraph'
import { forwardGrid } from '../../engine/forward'
import { layerSpecs, type Activation, type Dataset, type OptimizerKind } from '../../engine/types'
import { snapshotMatches } from '../../store/view'

const ACTS: { value: Activation; label: string }[] = [
  { value: 'relu', label: 'ReLU' },
  { value: 'tanh', label: 'tanh' },
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'gelu', label: 'GELU' },
]
const OPTS: { value: OptimizerKind; label: string }[] = [
  { value: 'sgd', label: 'SGD' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'adam', label: 'Adam' },
]
const lrFmt = (v: number) => (v >= 0.1 ? v.toFixed(2) : v.toPrecision(2))

const TONE = { critical: 'pos', warn: 'warn', info: 'ok' } as const

export function RunColumn({
  name,
  color,
  run,
  train,
  test,
}: {
  name: string
  color: string
  run: RunHandle
  train: Dataset
  test: Dataset
}) {
  const { useRun, viewSnapshot } = run
  const config = useRun((s) => s.config)
  const setConfig = useRun((s) => s.setConfig)
  const applyFix = useRun((s) => s.applyFix)
  const diagnoses = useRun((s) => s.diagnoses)
  const backend = useRun((s) => s.backend)
  const error = useRun((s) => s.error)
  const version = useRun((s) => s.version)
  void version

  const specs = useMemo(() => layerSpecs(config), [config])
  const snap = viewSnapshot(null)
  const view = snap && snapshotMatches(specs, snap) ? snap : undefined
  const grid = useMemo(
    () => (view ? forwardGrid(specs, view.weights, GRID, EXTENT) : null),
    [view, specs],
  )
  const setLayer = (i: number, patch: Partial<(typeof config.layers)[number]>) =>
    setConfig({ layers: config.layers.map((l, k) => (k === i ? { ...l, ...patch } : l)) })

  return (
    <section
      aria-label={`Run ${name}`}
      className="min-w-0 space-y-3 rounded-2xl border bg-panel/40 p-2"
      style={{ borderColor: color }}
    >
      <header className="flex items-center justify-between px-1">
        <h2 className="text-sm font-bold" style={{ color }}>
          Run {name}
        </h2>
        {backend && <Badge tone={backend === 'webgpu' ? 'accent' : 'ok'}>{backend}</Badge>}
      </header>

      <Panel
        title="Settings"
        info="Change anything here: the run restarts with the new settings (learning rate, L2 and batch size apply live). Data settings are shared between A and B."
      >
        <div className="space-y-2.5">
          <div className="grid grid-cols-2 gap-2">
            <Select
              label="Optimizer"
              value={config.optimizer}
              options={OPTS}
              onChange={(v) => setConfig({ optimizer: v })}
            />
            <Slider
              label="Learning rate"
              value={Math.log10(config.lr)}
              min={-3.5}
              max={0.47}
              step={0.02}
              onChange={(v) => setConfig({ lr: Number(Math.pow(10, v).toPrecision(2)) })}
              format={() => lrFmt(config.lr)}
            />
            <Slider
              label="L2"
              value={config.l2}
              min={0}
              max={0.05}
              step={0.001}
              onChange={(v) => setConfig({ l2: v })}
              format={(v) => v.toFixed(3)}
            />
            <Slider
              label="Batch size"
              value={config.batchSize}
              min={1}
              max={256}
              step={1}
              onChange={(v) => setConfig({ batchSize: v })}
            />
          </div>
          {config.layers.map((l, i) => (
            <div
              key={i}
              className="grid grid-cols-[1fr_1fr_auto] items-end gap-2 rounded-lg bg-panel2 p-2"
            >
              <Slider
                label={`Layer ${i + 1} neurons`}
                value={l.units}
                min={1}
                max={16}
                step={1}
                onChange={(v) => setLayer(i, { units: v })}
              />
              <Select
                label="Activation"
                value={l.activation}
                options={ACTS}
                onChange={(v) => setLayer(i, { activation: v })}
              />
              <button
                type="button"
                aria-label={`Remove layer ${i + 1} of run ${name}`}
                disabled={config.layers.length <= 1}
                onClick={() => setConfig({ layers: config.layers.filter((_, k) => k !== i) })}
                className="pb-1.5 text-xs text-muted hover:text-pos disabled:opacity-30"
              >
                ✕
              </button>
            </div>
          ))}
          <Btn
            className="w-full !py-1 text-xs"
            disabled={config.layers.length >= 5}
            onClick={() =>
              setConfig({
                layers: [
                  ...config.layers,
                  { units: 6, activation: config.layers.at(-1)?.activation ?? 'relu' },
                ],
              })
            }
          >
            + Add layer
          </Btn>
        </div>
      </Panel>

      {error && (
        <p role="alert" className="text-xs text-pos">
          {error}
        </p>
      )}

      <div className="grid grid-cols-3 gap-1.5 text-center">
        {[
          ['step', view ? String(view.step) : '–'],
          ['train loss', view ? view.trainLoss.toFixed(3) : '–'],
          ['test acc', view ? `${(view.testAcc * 100).toFixed(1)}%` : '–'],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border border-line bg-panel px-2 py-1">
            <div className="text-[10px] uppercase tracking-wider text-muted">{k}</div>
            <div className="num text-sm font-semibold">{v}</div>
          </div>
        ))}
      </div>

      <Panel
        title="Decision boundary"
        info="What run's network currently predicts at every point. Compare the shapes the two networks carve out."
      >
        <DecisionBoundary
          field={grid ? grid[grid.length - 1] : undefined}
          train={train}
          test={test}
          size={280}
        />
      </Panel>

      <Panel
        title="Network"
        info="Weights as lines (red positive, blue negative) and average neuron activity as colour. Dashed × circles are dead ReLUs."
      >
        <NetworkGraph
          config={config}
          snap={view}
          selected={null}
          onSelect={() => {}}
          height={240}
        />
      </Panel>

      <Panel
        title="Model Doctor"
        info="The same deterministic rules as in the playground, run on this experiment's telemetry."
      >
        <div aria-live="polite" className="space-y-1.5">
          {diagnoses.length === 0 && (
            <p className="text-xs text-muted">Watching… (needs a few dozen steps)</p>
          )}
          {diagnoses.map((d) => (
            <div key={d.id} className="rounded-lg border border-line bg-panel2 p-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold">{d.title}</span>
                <Badge tone={TONE[d.severity]}>{d.severity}</Badge>
              </div>
              <p className="mt-0.5 text-[11px] text-muted">
                step <span className="num text-ink">{d.evidence.step}</span>: {d.evidence.detail}
              </p>
              {d.fix && (
                <Btn
                  variant="primary"
                  className="mt-1.5 !py-0.5 text-[11px]"
                  onClick={() => applyFix(d.fix!)}
                >
                  ✨ {d.fix.label}
                </Btn>
              )}
            </div>
          ))}
        </div>
      </Panel>
    </section>
  )
}
