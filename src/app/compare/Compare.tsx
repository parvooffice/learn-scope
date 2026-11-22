import { useEffect, useMemo, useState } from 'react'
import { createRun, PRESETS, type RunHandle } from './runs'
import { RunColumn } from './RunColumn'
import { Btn, Panel, Select, Slider } from '../../components/ui'
import { LineChart, type Series } from '../../viz/LineChart'
import { generateDataset, splitDataset, DATASET_LABELS } from '../../engine/datasets/toy'
import { sanitizeConfig, decodeConfig, encodeConfig } from '../../lib/urlConfig'
import { replaceQuery, useRoute } from '../../lib/router'
import { useShortcuts } from '../../lib/shortcuts'
import type { DatasetKind, PlaygroundConfig } from '../../engine/types'

const A_COLOR = '#2dd4bf'
const B_COLOR = '#a78bfa'
const DATASETS = (Object.keys(DATASET_LABELS) as DatasetKind[]).map((k) => ({
  value: k,
  label: DATASET_LABELS[k],
}))

export default function Compare() {
  const { query } = useRoute()
  const [runs, setRuns] = useState<{ a: RunHandle; b: RunHandle } | null>(null)
  const [presetId, setPresetId] = useState('optimizer')

  // create the two runs (two workers) on mount, tear them down on unmount
  useEffect(() => {
    const preset = PRESETS[0]
    const ca = query.get('a') ? decodeConfig(query.get('a')!) : null
    const cb = query.get('b') ? decodeConfig(query.get('b')!) : null
    const a = createRun(ca ?? { ...preset.shared, ...preset.a })
    const b = createRun(cb ?? { ...preset.shared, ...preset.b })
    setRuns({ a, b })
    return () => {
      a.terminate()
      b.terminate()
      setRuns(null)
    }
    // the query is only read once, at mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (!runs) return <div className="p-6 text-sm text-muted">Starting two training workers…</div>
  return <CompareBody a={runs.a} b={runs.b} presetId={presetId} setPresetId={setPresetId} />
}

function CompareBody({
  a,
  b,
  presetId,
  setPresetId,
}: {
  a: RunHandle
  b: RunHandle
  presetId: string
  setPresetId: (id: string) => void
}) {
  const configA = a.useRun((s) => s.config)
  const configB = b.useRun((s) => s.config)
  const runningA = a.useRun((s) => s.running)
  const runningB = b.useRun((s) => s.running)
  const readyA = a.useRun((s) => s.ready)
  const readyB = b.useRun((s) => s.ready)
  const speed = a.useRun((s) => s.speed)
  const verA = a.useRun((s) => s.version)
  const verB = b.useRun((s) => s.version)
  const diagA = a.useRun((s) => s.diagnoses)
  const diagB = b.useRun((s) => s.diagnoses)
  const running = runningA || runningB
  const ready = readyA && readyB
  void verA
  void verB

  // data is shared: both runs always use the same dataset, split and seed
  const shared = {
    dataset: configA.dataset,
    noise: configA.noise,
    samples: configA.samples,
    testSplit: configA.testSplit,
    seed: configA.seed,
  }
  const { train, test } = useMemo(() => {
    const ds = generateDataset(shared.dataset, shared.samples, shared.noise, shared.seed)
    return splitDataset(ds, shared.testSplit, shared.seed)
  }, [shared.dataset, shared.samples, shared.noise, shared.seed, shared.testSplit])

  const setShared = (patch: Partial<PlaygroundConfig>) => {
    a.useRun.getState().setConfig(patch)
    b.useRun.getState().setConfig(patch)
  }
  const playBoth = () => {
    a.useRun.getState().play()
    b.useRun.getState().play()
  }
  const pauseBoth = () => {
    a.useRun.getState().pause()
    b.useRun.getState().pause()
  }
  const toggle = () => (running ? pauseBoth() : playBoth())
  const stepBoth = (n: number) => {
    a.useRun.getState().step(n)
    b.useRun.getState().step(n)
  }
  const resetBoth = () => {
    a.useRun.getState().reset()
    b.useRun.getState().reset()
  }
  const loadPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)
    if (!p) return
    setPresetId(id)
    const base = { ...p.shared }
    a.useRun.getState().replaceConfig(sanitizeConfig({ ...configA, ...base, ...p.a }))
    b.useRun.getState().replaceConfig(sanitizeConfig({ ...configB, ...base, ...p.b }))
    playBoth()
  }

  // start training automatically once both workers are ready
  const [started, setStarted] = useState(false)
  useEffect(() => {
    if (ready && !started) {
      setStarted(true)
      playBoth()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, started])

  // shareable URL for both experiments
  useEffect(() => {
    const t = setTimeout(
      () => replaceQuery('compare', `a=${encodeConfig(configA)}&b=${encodeConfig(configB)}`),
      400,
    )
    return () => clearTimeout(t)
  }, [configA, configB])

  useShortcuts([
    { key: ' ', run: toggle },
    { key: 'ArrowRight', run: () => stepBoth(1) },
    { key: 'r', run: resetBoth },
  ])

  const snapsA = a.history.toArray()
  const snapsB = b.history.toArray()
  const series = (key: 'trainLoss' | 'testLoss' | 'trainAcc' | 'testAcc'): Series[] => [
    {
      label: 'A train',
      color: A_COLOR,
      x: snapsA.map((s) => s.step),
      y: snapsA.map((s) => s[key]),
    },
    {
      label: 'B train',
      color: B_COLOR,
      x: snapsB.map((s) => s.step),
      y: snapsB.map((s) => s[key]),
    },
  ]
  const both = (train: 'trainLoss' | 'trainAcc', test: 'testLoss' | 'testAcc'): Series[] => [
    ...series(train),
    {
      label: 'A test',
      color: A_COLOR,
      dashed: true,
      x: snapsA.map((s) => s.step),
      y: snapsA.map((s) => s[test]),
    },
    {
      label: 'B test',
      color: B_COLOR,
      dashed: true,
      x: snapsB.map((s) => s.step),
      y: snapsB.map((s) => s[test]),
    },
  ]

  const lastA = snapsA.at(-1)
  const lastB = snapsB.at(-1)
  const preset = PRESETS.find((p) => p.id === presetId)

  const winner = (va?: number, vb?: number, lowerBetter = false) => {
    if (va === undefined || vb === undefined || Math.abs(va - vb) < 1e-9) return ''
    return (lowerBetter ? va < vb : va > vb) ? 'A' : 'B'
  }
  const rows: [string, string, string, string][] = [
    ['Steps', lastA ? String(lastA.step) : '–', lastB ? String(lastB.step) : '–', ''],
    [
      'Train loss',
      lastA ? lastA.trainLoss.toFixed(4) : '–',
      lastB ? lastB.trainLoss.toFixed(4) : '–',
      winner(lastA?.trainLoss, lastB?.trainLoss, true),
    ],
    [
      'Test loss',
      lastA ? lastA.testLoss.toFixed(4) : '–',
      lastB ? lastB.testLoss.toFixed(4) : '–',
      winner(lastA?.testLoss, lastB?.testLoss, true),
    ],
    [
      'Test accuracy',
      lastA ? `${(lastA.testAcc * 100).toFixed(1)}%` : '–',
      lastB ? `${(lastB.testAcc * 100).toFixed(1)}%` : '–',
      winner(lastA?.testAcc, lastB?.testAcc),
    ],
    [
      'Doctor findings',
      String(diagA.filter((d) => d.id !== 'healthy').length),
      String(diagB.filter((d) => d.id !== 'healthy').length),
      '',
    ],
  ]

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-[1500px] space-y-3 p-3">
        <div>
          <h1 className="text-xl font-semibold">Compare two experiments</h1>
          <p className="text-sm text-muted">
            Two networks train at the same time on the <b>same data</b> (each in its own worker).
            Change anything in A or B and watch how the curves, boundaries and Doctor findings
            differ.
          </p>
        </div>

        <div className="flex flex-wrap items-end gap-2 rounded-xl border border-line bg-panel p-3">
          <Btn
            variant="primary"
            onClick={toggle}
            disabled={!ready}
            ariaLabel={running ? 'Pause both runs' : 'Play both runs'}
            title="Space"
          >
            {running ? '⏸ Pause both' : '▶ Play both'}
          </Btn>
          <Btn onClick={() => stepBoth(1)} disabled={!ready}>
            Step
          </Btn>
          <Btn onClick={() => stepBoth(10)} disabled={!ready}>
            Step ×10
          </Btn>
          <Btn onClick={resetBoth} disabled={!ready} title="R">
            ↺ Reset both
          </Btn>
          <div className="w-32">
            <Slider
              label="Speed"
              value={speed}
              min={1}
              max={32}
              step={1}
              onChange={(v) => {
                a.useRun.getState().setSpeed(v)
                b.useRun.getState().setSpeed(v)
              }}
              format={(v) => `${v}×`}
            />
          </div>
          <div className="ml-auto flex flex-wrap items-end gap-2">
            <div className="w-56">
              <Select
                label="Preset comparison"
                value={presetId}
                options={PRESETS.map((p) => ({ value: p.id, label: p.label }))}
                onChange={loadPreset}
              />
            </div>
          </div>
          {preset && <p className="w-full text-xs text-muted">💡 {preset.blurb}</p>}
        </div>

        <Panel
          title="Shared data"
          info="Both experiments train on exactly the same points (same dataset, noise, split and random seed), so any difference you see comes from the settings you changed, not from luck of the data."
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <Select
              label="Dataset"
              value={shared.dataset}
              options={DATASETS}
              onChange={(v) => setShared({ dataset: v })}
            />
            <Slider
              label="Noise"
              value={shared.noise}
              min={0}
              max={0.5}
              step={0.01}
              onChange={(v) => setShared({ noise: v })}
              format={(v) => v.toFixed(2)}
            />
            <Slider
              label="Samples"
              value={shared.samples}
              min={100}
              max={1000}
              step={50}
              onChange={(v) => setShared({ samples: v })}
            />
            <Slider
              label="Test split"
              value={shared.testSplit}
              min={0.1}
              max={0.5}
              step={0.05}
              onChange={(v) => setShared({ testSplit: v })}
              format={(v) => `${Math.round(v * 100)}%`}
            />
            <div className="flex items-end">
              <Btn
                className="w-full"
                onClick={() => setShared({ seed: Math.floor(Math.random() * 99999) })}
              >
                🎲 New seed
              </Btn>
            </div>
          </div>
        </Panel>

        <div className="grid gap-3 lg:grid-cols-2">
          <Panel
            title="Loss (A vs B)"
            info="Training loss (solid) and test loss (dashed) for both runs on one chart. Teal is A, violet is B. Lower is better; a dashed line pulling away above its solid twin means that run is overfitting."
          >
            <LineChart
              series={both('trainLoss', 'testLoss')}
              label="Loss of run A and run B over training steps"
              yMin={0}
              height={190}
            />
          </Panel>
          <Panel
            title="Accuracy (A vs B)"
            info="The fraction of points classified correctly, training (solid) and test (dashed), for both runs."
          >
            <LineChart
              series={both('trainAcc', 'testAcc')}
              label="Accuracy of run A and run B over training steps"
              yMin={0}
              yMax={1}
              height={190}
              yFormat={(v) => `${Math.round(v * 100)}%`}
            />
          </Panel>
        </div>

        <Panel
          title="Scoreboard"
          info="The latest numbers from each run. A marker shows which run is ahead on each metric."
        >
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted">
                <th className="py-1 font-medium">Metric</th>
                <th className="py-1 font-medium" style={{ color: A_COLOR }}>
                  Run A
                </th>
                <th className="py-1 font-medium" style={{ color: B_COLOR }}>
                  Run B
                </th>
                <th className="py-1 font-medium">Ahead</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([k, va, vb, w]) => (
                <tr key={k} className="border-t border-line">
                  <td className="py-1 text-muted">{k}</td>
                  <td className="num py-1">{va}</td>
                  <td className="num py-1">{vb}</td>
                  <td
                    className="py-1 text-xs"
                    style={{ color: w === 'A' ? A_COLOR : w === 'B' ? B_COLOR : undefined }}
                  >
                    {w && `Run ${w}`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <div className="grid gap-3 lg:grid-cols-2">
          <RunColumn name="A" color={A_COLOR} run={a} train={train} test={test} />
          <RunColumn name="B" color={B_COLOR} run={b} train={train} test={test} />
        </div>
      </div>
    </div>
  )
}
