import { history, useRun } from '../../store/run'
import { useViewSnapshot } from '../../store/view'
import { Panel } from '../../components/ui'
import { LineChart, type Series } from '../../viz/LineChart'
import { Histogram } from '../../viz/Histogram'
import { SERIES } from '../../lib/colors'
import { layerSpecs } from '../../engine/types'

const fmtPct = (x: number) => `${(x * 100).toFixed(x < 0.1 && x > 0 ? 1 : 0)}%`

export function Charts() {
  const version = useRun((s) => s.version)
  const selectedStep = useRun((s) => s.selectedStep)
  const config = useRun((s) => s.config)
  const jumpToStep = useRun((s) => s.jumpToStep)
  void version
  const snaps = history.toArray()
  const view = useViewSnapshot()
  const steps = snaps.map((s) => s.step)
  const marker = selectedStep === null ? null : view?.step ?? null
  const specs = layerSpecs(config)

  const loss: Series[] = [
    { label: 'train', color: '#2dd4bf', x: steps, y: snaps.map((s) => s.trainLoss) },
    { label: 'test', color: '#f5b942', x: steps, y: snaps.map((s) => s.testLoss) },
  ]
  const acc: Series[] = [
    { label: 'train', color: '#2dd4bf', x: steps, y: snaps.map((s) => s.trainAcc) },
    { label: 'test', color: '#f5b942', x: steps, y: snaps.map((s) => s.testAcc) },
  ]
  const nL = specs.length
  const grad: Series[] = Array.from({ length: nL }, (_, l) => ({
    label: l === nL - 1 ? 'out' : `L${l + 1}`,
    color: SERIES[l % SERIES.length],
    x: steps,
    y: snaps.map((s) => s.layers[l]?.gNorm ?? NaN),
  }))

  return (
    <div className="space-y-3">
      <Panel
        chart="loss"
        title="Loss"
        info="Loss is a single number for “how wrong the network is” (lower is better). The teal line is the loss on the training points, the amber line on held-out test points. If the lines drift apart, the network is memorising. Click the chart to time-travel to that step."
      >
        <LineChart series={loss} label="Training and test loss over steps" marker={marker} onPickX={jumpToStep} yMin={0} />
      </Panel>

      <Panel
        title="Accuracy"
        info="The fraction of points the network classifies correctly, for training points (teal) and test points (amber). 50% is a coin flip on these balanced datasets."
      >
        <LineChart series={acc} label="Training and test accuracy over steps" marker={marker} onPickX={jumpToStep} yMin={0} yMax={1} yFormat={(v) => `${Math.round(v * 100)}%`} height={120} />
      </Panel>

      <Panel
        chart="grad"
        title="Gradient size per layer"
        info="The gradient tells each weight which way to move to reduce the loss. This chart shows how large the gradient is in each layer (log scale). If the early layers (L1…) sit far below the output layer, gradients are vanishing and those layers barely learn."
      >
        <LineChart series={grad} label="Gradient norm per layer over steps, log scale" log marker={marker} onPickX={jumpToStep} />
      </Panel>

      <Panel
        chart="weights"
        title="Weight distributions"
        info="A histogram of every weight in each layer at the selected step. Training starts with small random values and spreads them out. If a histogram stretches very wide, weights are growing large; teal bars to the left of the dashed line are negative weights."
      >
        {view ? (
          <div className="space-y-2">
            {view.layers.map((l, i) => (
              <div key={i}>
                <div className="flex justify-between text-[11px] text-muted">
                  <span>{i === nL - 1 ? 'output' : `layer ${i + 1}`}</span>
                  <span className="num">
                    μ {l.wMean.toFixed(2)} · σ {l.wStd.toFixed(2)}
                  </span>
                </div>
                <Histogram hist={l.wHist} color={SERIES[i % SERIES.length]} label={`Weight histogram for layer ${i + 1}`} />
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted">Waiting for the first step…</p>
        )}
      </Panel>

      <Panel
        chart="activations"
        title="Activations per layer"
        info="Each neuron's output on the training points. A healthy layer has a spread of values. For ReLU layers, a tall bar at 0 and a high “dead” percentage means many neurons have stopped firing for good. For tanh/sigmoid, bars piled at the far ends mean the neurons are saturated."
      >
        {view ? (
          <div className="space-y-2">
            {view.layers.map((l, i) => {
              const act = specs[i].activation
              return (
                <div key={i}>
                  <div className="flex justify-between text-[11px] text-muted">
                    <span>
                      {i === nL - 1 ? 'output' : `layer ${i + 1}`} · {act}
                    </span>
                    <span className="num">
                      {act === 'relu' || act === 'gelu' ? (
                        <span className={l.deadPct > 0.5 ? 'text-pos' : ''}>dead {fmtPct(l.deadPct)}</span>
                      ) : (
                        <span className={l.satPct > 0.6 ? 'text-warn' : ''}>saturated {fmtPct(l.satPct)}</span>
                      )}
                    </span>
                  </div>
                  <Histogram hist={l.aHist} color={SERIES[i % SERIES.length]} label={`Activation histogram for layer ${i + 1}`} zeroLine={false} />
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-xs text-muted">Waiting for the first step…</p>
        )}
      </Panel>
    </div>
  )
}
