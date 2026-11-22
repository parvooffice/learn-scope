import { useRun } from '../../store/run'
import { Btn, Panel, Select, Slider } from '../../components/ui'
import { DATASET_LABELS } from '../../engine/datasets/toy'
import type { Activation, DatasetKind, OptimizerKind } from '../../engine/types'

const DATASET_OPTS = (Object.keys(DATASET_LABELS) as DatasetKind[]).map((k) => ({
  value: k,
  label: DATASET_LABELS[k],
}))
const ACT_OPTS: { value: Activation; label: string }[] = [
  { value: 'relu', label: 'ReLU' },
  { value: 'tanh', label: 'tanh' },
  { value: 'sigmoid', label: 'Sigmoid' },
  { value: 'gelu', label: 'GELU' },
]
const OPT_OPTS: { value: OptimizerKind; label: string }[] = [
  { value: 'sgd', label: 'SGD' },
  { value: 'momentum', label: 'SGD + momentum' },
  { value: 'adam', label: 'Adam' },
]

const lrFormat = (v: number) =>
  v >= 0.1 ? v.toFixed(2) : v >= 0.01 ? v.toFixed(3) : v.toPrecision(2)

export function ConfigPanel() {
  const config = useRun((s) => s.config)
  const setConfig = useRun((s) => s.setConfig)

  const setLayer = (i: number, patch: Partial<(typeof config.layers)[number]>) =>
    setConfig({ layers: config.layers.map((l, k) => (k === i ? { ...l, ...patch } : l)) })

  return (
    <div className="space-y-3">
      <Panel
        title="Data"
        info="The 2-D toy datasets the network learns to separate: each point is either class 0 (white dot) or class 1 (black dot). Noise blurs the classes. Solid dots are used for training; rings are held out as the test set to check that the network generalizes."
      >
        <div className="space-y-3">
          <Select
            label="Dataset"
            value={config.dataset}
            options={DATASET_OPTS}
            onChange={(v) => setConfig({ dataset: v })}
          />
          <Slider
            label="Noise"
            value={config.noise}
            min={0}
            max={0.5}
            step={0.01}
            onChange={(v) => setConfig({ noise: v })}
            format={(v) => v.toFixed(2)}
          />
          <Slider
            label="Samples"
            value={config.samples}
            min={100}
            max={1000}
            step={50}
            onChange={(v) => setConfig({ samples: v })}
          />
          <Slider
            label="Test split"
            value={config.testSplit}
            min={0.1}
            max={0.5}
            step={0.05}
            onChange={(v) => setConfig({ testSplit: v })}
            format={(v) => `${Math.round(v * 100)}%`}
          />
          <div className="flex gap-2">
            <Btn
              onClick={() => setConfig({ seed: Math.floor(Math.random() * 99999) })}
              title="Different random data and weight initialisation"
              className="flex-1"
            >
              🎲 New random seed
            </Btn>
          </div>
        </div>
      </Panel>

      <Panel
        title="Model"
        info="The network is a stack of dense layers. Each hidden layer has some neurons and an activation function, which bends straight lines into curves. The last layer is always one sigmoid neuron that outputs the probability of class 1. Changing the architecture restarts training."
      >
        <div className="space-y-3">
          {config.layers.map((l, i) => (
            <div key={i} className="rounded-lg border border-line bg-panel2 p-2">
              <div className="mb-1.5 flex items-center justify-between">
                <span className="text-xs font-medium text-ink">Hidden layer {i + 1}</span>
                <button
                  type="button"
                  aria-label={`Remove hidden layer ${i + 1}`}
                  disabled={config.layers.length <= 1}
                  onClick={() => setConfig({ layers: config.layers.filter((_, k) => k !== i) })}
                  className="rounded px-1.5 text-xs text-muted hover:text-pos disabled:opacity-30"
                >
                  remove
                </button>
              </div>
              <div className="space-y-2">
                <Slider
                  label="Neurons"
                  value={l.units}
                  min={1}
                  max={16}
                  step={1}
                  onChange={(v) => setLayer(i, { units: v })}
                />
                <Select
                  label="Activation"
                  value={l.activation}
                  options={ACT_OPTS}
                  onChange={(v) => setLayer(i, { activation: v })}
                />
              </div>
            </div>
          ))}
          <Btn
            className="w-full"
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
            + Add hidden layer
          </Btn>
        </div>
      </Panel>

      <Panel
        title="Training"
        info="How the network learns. The optimizer decides how to use each gradient, the learning rate is the step size, and the batch size is how many points are used for each step. L2 regularization gently pulls weights toward zero to fight overfitting. Learning rate, L2 and batch size apply instantly; the rest restarts training."
      >
        <div className="space-y-3">
          <Select
            label="Optimizer"
            value={config.optimizer}
            options={OPT_OPTS}
            onChange={(v) => setConfig({ optimizer: v })}
          />
          <Slider
            label="Learning rate"
            value={Math.log10(config.lr)}
            min={-3.5}
            max={0.47}
            step={0.02}
            onChange={(v) => setConfig({ lr: Number(Math.pow(10, v).toPrecision(2)) })}
            format={() => lrFormat(config.lr)}
            hint="Step size. Too small: slow. Too large: unstable."
          />
          <Slider
            label="Batch size"
            value={config.batchSize}
            min={1}
            max={256}
            step={1}
            onChange={(v) => setConfig({ batchSize: v })}
          />
          <Slider
            label="L2 regularization"
            value={config.l2}
            min={0}
            max={0.05}
            step={0.001}
            onChange={(v) => setConfig({ l2: v })}
            format={(v) => v.toFixed(3)}
          />
        </div>
      </Panel>
    </div>
  )
}
