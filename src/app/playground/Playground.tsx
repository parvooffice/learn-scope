import { useEffect, useMemo, useRef, useState } from 'react'
import { useRun } from '../../store/run'
import { useViewSnapshot } from '../../store/view'
import { ConfigPanel } from './ConfigPanel'
import { Toolbar } from './Toolbar'
import { Charts } from './Charts'
import { DoctorPanel } from './DoctorPanel'
import { Timeline } from './Timeline'
import { Panel } from '../../components/ui'
import { DecisionBoundary, EXTENT, GRID } from '../../viz/DecisionBoundary'
import { NetworkGraph, type NeuronRef } from '../../viz/NetworkGraph'
import { NeuronPreview } from '../../viz/NeuronPreview'
import { LossLandscape } from './LossLandscape'
import { generateDataset, splitDataset } from '../../engine/datasets/toy'
import { forwardGrid } from '../../engine/forward'
import { layerSpecs } from '../../engine/types'
import { decodeConfig, encodeConfig } from '../../lib/urlConfig'
import { replaceQuery, useRoute } from '../../lib/router'
import { useShortcuts } from '../../lib/shortcuts'

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-line bg-panel px-3 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div
        data-testid={`stat-${label.replace(/\s+/g, '-')}`}
        className={`num text-sm font-semibold ${tone ?? 'text-ink'}`}
      >
        {value}
      </div>
    </div>
  )
}

export default function Playground() {
  const { query } = useRoute()
  const boot = useRun((s) => s.boot)
  const config = useRun((s) => s.config)
  const ready = useRun((s) => s.ready)
  const error = useRun((s) => s.error)
  const backend = useRun((s) => s.backend)
  const notes = useRun((s) => s.notes)
  const version = useRun((s) => s.version)
  const toggle = useRun((s) => s.toggle)
  const step = useRun((s) => s.step)
  const reset = useRun((s) => s.reset)
  const nudge = useRun((s) => s.nudge)
  const goLive = useRun((s) => s.goLive)
  const selectedStep = useRun((s) => s.selectedStep)
  const view = useViewSnapshot()
  const [neuron, setNeuron] = useState<NeuronRef | null>(null)
  const autoplayed = useRef(false)

  // load a shared config from the URL before the worker is created
  const loaded = useRef(false)
  if (!loaded.current) {
    loaded.current = true
    const c = query.get('c')
    const decoded = c ? decodeConfig(c) : null
    if (decoded) useRun.setState({ config: decoded })
  }

  useEffect(() => {
    boot()
  }, [boot])

  // "Try it" opens the playground already training
  useEffect(() => {
    if (ready && version > 0 && !autoplayed.current) {
      autoplayed.current = true
      if (query.get('play') !== '0') useRun.getState().play()
    }
  }, [ready, version, query])

  // keep the URL in sync so any state is shareable
  useEffect(() => {
    const t = setTimeout(() => replaceQuery('playground', `c=${encodeConfig(config)}`), 400)
    return () => clearTimeout(t)
  }, [config])

  useShortcuts([
    { key: ' ', run: toggle },
    { key: 'ArrowRight', run: () => (selectedStep === null ? step(1) : nudge(1)) },
    { key: 'ArrowRight', shift: true, run: () => step(10) },
    { key: 'ArrowLeft', run: () => nudge(-1) },
    { key: 'r', run: reset },
    { key: 'l', run: goLive },
  ])

  const specs = useMemo(() => layerSpecs(config), [config])
  const { train, test } = useMemo(() => {
    const ds = generateDataset(config.dataset, config.samples, config.noise, config.seed)
    return splitDataset(ds, config.testSplit, config.seed)
  }, [config.dataset, config.samples, config.noise, config.seed, config.testSplit])

  const grid = useMemo(
    () => (view ? forwardGrid(specs, view.weights, GRID, EXTENT) : null),
    [view, specs],
  )
  const field = grid ? grid[grid.length - 1] : undefined
  const validNeuron =
    neuron && neuron.layer < specs.length - 1 && neuron.unit < specs[neuron.layer].nout
      ? neuron
      : null

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-3 md:grid-cols-2 lg:grid-cols-[290px_minmax(0,1fr)_390px] lg:overflow-hidden scroll-thin">
        <aside
          className="order-2 min-h-0 md:order-2 lg:order-1 lg:overflow-y-auto scroll-thin"
          aria-label="Configuration"
        >
          <ConfigPanel />
        </aside>

        <main
          className="order-1 space-y-3 md:order-1 md:col-span-2 lg:order-2 lg:col-span-1 lg:min-h-0 lg:overflow-y-auto scroll-thin"
          aria-label="Network and decision boundary"
        >
          <Toolbar />
          {error && (
            <div
              role="alert"
              className="rounded-lg border border-pos/50 bg-panel p-3 text-sm text-pos"
            >
              Training error: {error}
            </div>
          )}
          {!ready && !error && (
            <div className="rounded-lg border border-line bg-panel p-3 text-sm text-muted">
              Loading TensorFlow.js and selecting the fastest backend…
            </div>
          )}
          {ready && backend && backend !== 'webgpu' && (
            <div
              className="rounded-lg border border-warn/40 bg-panel p-3 text-xs text-muted"
              role="status"
            >
              <b className="text-warn">Heads up:</b> WebGPU isn't available here, so training runs
              on{' '}
              <b className="text-ink">
                {backend === 'webgl' ? 'WebGL (still on the GPU)' : 'the CPU'}
              </b>
              . Everything works, just possibly slower. Chrome or Edge with WebGPU gives the full
              experience.
              {notes.length > 0 && (
                <span className="block pt-1 opacity-80">{notes.join(' · ')}</span>
              )}
            </div>
          )}

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-6">
            <Stat label="step" value={String(view?.step ?? 0)} />
            <Stat
              label="train loss"
              value={view ? view.trainLoss.toFixed(4) : '–'}
              tone="text-accent"
            />
            <Stat
              label="test loss"
              value={view ? view.testLoss.toFixed(4) : '–'}
              tone="text-warn"
            />
            <Stat label="train acc" value={view ? `${(view.trainAcc * 100).toFixed(1)}%` : '–'} />
            <Stat label="test acc" value={view ? `${(view.testAcc * 100).toFixed(1)}%` : '–'} />
            <Stat label="learning rate" value={String(view?.lr ?? config.lr)} />
          </div>

          <div className="grid gap-3 xl:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
            <Panel
              chart="network"
              title="Decision boundary"
              info="The heat map is the network's prediction at every point of the plane: purple = confident class 0, yellow = confident class 1, and the boundary is where the colour switches. Dots are training points; rings are test points the network never trained on."
            >
              <DecisionBoundary field={field} train={train} test={test} />
              <p className="mt-1 text-[11px] text-muted">
                ● class 1 &nbsp; ○ class 0 &nbsp; (rings = test set)
              </p>
            </Panel>

            <div className="space-y-3">
              <Panel
                title="Network"
                info="Circles are neurons and lines are weights. Red lines are positive weights, blue lines are negative, and thicker means larger. A neuron's fill colour is how active it is on average; a neuron drawn dashed with × is a dead ReLU. Click any hidden neuron to see what it responds to."
              >
                <NetworkGraph
                  config={config}
                  snap={view}
                  selected={validNeuron}
                  onSelect={setNeuron}
                />
              </Panel>
              {validNeuron && grid && (
                <Panel
                  title={`Neuron · layer ${validNeuron.layer + 1} unit ${validNeuron.unit + 1}`}
                  info="This neuron's own 'decision boundary': how strongly it fires at each point of the input plane. Each neuron carves out one simple shape (a slanted band, a corner); the layers after it combine those shapes into the final boundary."
                  actions={
                    <button
                      type="button"
                      className="text-xs text-muted hover:text-ink"
                      onClick={() => setNeuron(null)}
                    >
                      close
                    </button>
                  }
                >
                  <NeuronPreview
                    grid={grid[validNeuron.layer]}
                    nout={specs[validNeuron.layer].nout}
                    unit={validNeuron.unit}
                    size={150}
                  />
                </Panel>
              )}
            </div>
          </div>
          <LossLandscape specs={specs} train={train} view={view} />
        </main>

        <aside
          className="order-3 min-h-0 space-y-3 lg:overflow-y-auto scroll-thin"
          aria-label="Charts and Model Doctor"
        >
          <DoctorPanel />
          <Charts />
        </aside>
      </div>
      <Timeline />
    </div>
  )
}
