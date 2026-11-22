import { useEffect, useMemo } from 'react'
import { mnistHistory, testPixels, useMnist } from '../../store/mnist'
import { Badge, Btn, Panel, Select, Slider } from '../../components/ui'
import { LineChart } from '../../viz/LineChart'
import { MapGrid } from '../../viz/MapGrid'
import { DigitCanvas } from '../../viz/DigitCanvas'
import { ConfusionMatrix } from '../../viz/ConfusionMatrix'
import { DrawPad } from '../../viz/DrawPad'
import { AutoCanvas } from '../../components/AutoCanvas'
import { viridis } from '../../lib/colors'
import { useShortcuts } from '../../lib/shortcuts'
import type { MnistModelKind } from '../../worker/mnistProtocol'

function ProbBars({ probs, truth }: { probs: number[]; truth?: number }) {
  const top = probs.length ? probs.indexOf(Math.max(...probs)) : -1
  return (
    <div className="space-y-1" role="list" aria-label="Predicted probability for each digit">
      {Array.from({ length: 10 }, (_, d) => {
        const p = probs[d] ?? 0
        return (
          <div key={d} role="listitem" className="flex items-center gap-2 text-xs">
            <span
              className={`w-3 text-right font-semibold ${d === top ? 'text-accent' : 'text-muted'}`}
            >
              {d}
            </span>
            <div className="h-3 flex-1 overflow-hidden rounded bg-panel2">
              <div
                className={`h-full rounded ${d === top ? 'bg-accent' : 'bg-muted/50'}`}
                style={{ width: `${Math.max(0.5, p * 100)}%`, transition: 'width 120ms linear' }}
              />
            </div>
            <span className="num w-12 text-right text-muted">{(p * 100).toFixed(1)}%</span>
            {truth === d && <span className="text-[10px] text-ok">truth</span>}
          </div>
        )
      })}
    </div>
  )
}

function HiddenStrip({ values }: { values: number[] }) {
  const max = Math.max(1e-6, ...values)
  const cols = 16
  return (
    <AutoCanvas
      height={Math.ceil(values.length / cols) * 14 + 2}
      label="Activations of the 128 neurons in the first hidden layer for this image"
      draw={(ctx, w) => {
        const cell = w / cols
        values.forEach((v, i) => {
          const [r, g, b] = viridis(v / max)
          ctx.fillStyle = `rgb(${r | 0},${g | 0},${b | 0})`
          ctx.fillRect((i % cols) * cell, Math.floor(i / cols) * 14, cell - 1, 13)
        })
      }}
    />
  )
}

export default function Mnist() {
  const s = useMnist()
  const { boot } = s
  useEffect(() => boot(), [boot])
  void s.version

  useShortcuts([
    { key: ' ', run: () => (s.running ? s.pause() : s.play()) },
    { key: 'ArrowRight', run: () => s.step(1) },
  ])

  const snap = s.snapshot
  const cnn = s.config.model === 'cnn'
  const busy = !s.loaded || !s.ready
  const truth =
    s.focusSource.kind === 'test' && s.data
      ? s.data.labels[s.data.nTrain + s.focusSource.index]
      : undefined
  const pixels = useMemo(() => {
    if (s.focusSource.kind === 'draw') return s.drawPixels
    return s.data ? testPixels(s.data, s.focusSource.index) : null
  }, [s.focusSource, s.drawPixels, s.data])

  const hist = mnistHistory
  const steps = hist.map((p) => p.step)
  const evals = hist.filter((p) => p.testLoss !== undefined)
  const probs = s.focus?.probs ?? []
  const pred = probs.length ? probs.indexOf(Math.max(...probs)) : null

  const budgetOk = s.epochSeconds !== null && s.epochSeconds <= 60

  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto grid max-w-[1500px] grid-cols-1 gap-3 p-3 md:grid-cols-2 xl:grid-cols-[280px_minmax(0,1fr)_380px]">
        {/* ----- left: controls ----- */}
        <aside className="space-y-3 md:col-span-2 xl:col-span-1" aria-label="MNIST controls">
          <Panel
            title="MNIST · real data on the GPU"
            info="MNIST is a classic dataset of handwritten digits (28×28 grayscale images). Here a subset of 10,000 training images and 2,000 test images is stored in your browser, and the network trains on your GPU in a Web Worker. Pick an MLP (plain dense layers) or a small CNN (convolutions + pooling)."
          >
            <div className="space-y-3">
              <Select
                label="Model"
                value={s.config.model}
                options={[
                  { value: 'cnn', label: 'Small CNN (conv → pool → conv → pool → dense)' },
                  { value: 'mlp', label: 'MLP (784 → 128 → 64 → 10)' },
                ]}
                onChange={(v: MnistModelKind) => s.setConfig({ model: v })}
              />
              <Slider
                label="Learning rate"
                value={Math.log10(s.config.lr)}
                min={-4}
                max={-1}
                step={0.05}
                onChange={(v) => s.setConfig({ lr: Number(Math.pow(10, v).toPrecision(2)) })}
                format={() => String(s.config.lr)}
              />
              <Select
                label="Batch size"
                value={String(s.config.batchSize)}
                options={[32, 64, 128, 256].map((n) => ({ value: String(n), label: String(n) }))}
                onChange={(v) => s.setConfig({ batchSize: Number(v) })}
              />
              <Slider
                label="Steps per frame"
                value={s.speed}
                min={1}
                max={16}
                step={1}
                onChange={s.setSpeed}
              />
              <div className="flex flex-wrap gap-2">
                <Btn variant="primary" disabled={busy} onClick={s.running ? s.pause : s.play}>
                  {s.running ? '⏸ Pause' : '▶ Train'}
                </Btn>
                <Btn disabled={busy} onClick={() => s.step(1)}>
                  Step
                </Btn>
                <Btn disabled={busy} onClick={s.reset}>
                  ↺ Reset
                </Btn>
              </div>
            </div>
          </Panel>

          <Panel
            title="Status"
            info="Where the data came from, which compute backend is training, and how fast it is. The performance budget for this app is: at least one epoch of the 10k images in under about 60 seconds on a mid-range laptop GPU."
          >
            <dl className="space-y-1.5 text-xs">
              <div className="flex justify-between gap-2">
                <dt className="text-muted">Data</dt>
                <dd className="text-right">{s.loadStatus}</dd>
              </div>
              <div className="flex justify-between gap-2">
                <dt className="text-muted">Backend</dt>
                <dd>
                  {s.backend ? (
                    <Badge
                      tone={
                        s.backend === 'webgpu' ? 'accent' : s.backend === 'webgl' ? 'ok' : 'warn'
                      }
                    >
                      {s.backend === 'webgpu'
                        ? 'GPU · WebGPU'
                        : s.backend === 'webgl'
                          ? 'GPU · WebGL'
                          : 'CPU fallback'}
                    </Badge>
                  ) : (
                    '…'
                  )}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">Parameters</dt>
                <dd className="num">{s.params.toLocaleString()}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">Step / epoch</dt>
                <dd className="num">
                  {snap?.step ?? 0} / {(snap?.epoch ?? 0).toFixed(2)}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">Throughput</dt>
                <dd className="num">
                  {snap && snap.imgPerSec > 0
                    ? `${Math.round(snap.imgPerSec).toLocaleString()} img/s`
                    : '–'}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-muted">Time per epoch</dt>
                <dd
                  className={`num ${s.epochSeconds === null ? '' : budgetOk ? 'text-ok' : 'text-warn'}`}
                >
                  {snap && snap.imgPerSec > 0 ? `${(10000 / snap.imgPerSec).toFixed(1)} s` : '–'}
                </dd>
              </div>
            </dl>
            {s.error && (
              <p role="alert" className="mt-2 text-xs text-pos">
                {s.error}
              </p>
            )}
            {s.backend && s.backend !== 'webgpu' && (
              <p className="mt-2 text-[11px] text-muted" role="status">
                <b className="text-warn">Note:</b> WebGPU unavailable; using{' '}
                {s.backend === 'webgl' ? 'WebGL' : 'the CPU'}. Training works but may be slower.
              </p>
            )}
          </Panel>

          <Panel
            title="Accuracy"
            info="The fraction of images classified correctly. Train accuracy is measured on the mini-batches being trained on; test accuracy on 2,000 held-out images the model never trains on."
          >
            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="rounded-lg border border-line bg-panel2 p-2">
                <div className="text-[10px] uppercase tracking-wider text-muted">train</div>
                <div className="num text-lg font-semibold text-accent">
                  {snap ? `${(snap.trainAcc * 100).toFixed(1)}%` : '–'}
                </div>
              </div>
              <div className="rounded-lg border border-line bg-panel2 p-2">
                <div className="text-[10px] uppercase tracking-wider text-muted">test</div>
                <div className="num text-lg font-semibold text-warn">
                  {s.lastEval ? `${(s.lastEval.testAcc * 100).toFixed(1)}%` : '–'}
                </div>
              </div>
            </div>
          </Panel>
        </aside>

        {/* ----- centre: look inside ----- */}
        <main className="min-w-0 space-y-3" aria-label="Inside the network">
          <div className="grid gap-3 lg:grid-cols-2">
            <Panel
              title="Test image"
              info="A test image the network is looking at, with its predicted probabilities. The feature maps below are for this exact image. Step through images with the buttons."
            >
              <div className="flex items-start gap-3">
                <DigitCanvas
                  pixels={s.focusSource.kind === 'test' ? pixels : null}
                  size={140}
                  label="The selected MNIST test image"
                />
                <div className="space-y-2 text-sm">
                  <div>
                    truth: <b className="num">{truth ?? '–'}</b>
                  </div>
                  <div>
                    predicted:{' '}
                    <b
                      className={`num ${pred !== null && truth !== undefined ? (pred === truth ? 'text-ok' : 'text-pos') : ''}`}
                    >
                      {pred ?? '–'}
                    </b>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    <Btn
                      className="!px-2 !py-1"
                      ariaLabel="Previous test image"
                      onClick={() =>
                        s.focusTest(
                          s.focusSource.kind === 'test' ? Math.max(0, s.focusSource.index - 1) : 0,
                        )
                      }
                    >
                      ‹
                    </Btn>
                    <Btn
                      className="!px-2 !py-1"
                      ariaLabel="Next test image"
                      onClick={() =>
                        s.focusTest(
                          s.focusSource.kind === 'test'
                            ? Math.min(1999, s.focusSource.index + 1)
                            : 0,
                        )
                      }
                    >
                      ›
                    </Btn>
                    <Btn
                      className="!px-2 !py-1"
                      onClick={() => s.focusTest(Math.floor(Math.random() * 2000))}
                    >
                      🎲 random
                    </Btn>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel
              title="Draw a digit"
              info="Draw any digit 0–9 with your mouse or finger. The picture is shrunk to 28×28 and centred like MNIST digits, then sent through the network on every stroke. Watch the predictions and feature maps change as you draw. Train the network first for good results!"
            >
              <div className="flex flex-wrap items-start gap-3">
                <DrawPad onPixels={s.focusDrawn} />
                {s.focusSource.kind === 'draw' && (
                  <DigitCanvas
                    pixels={pixels}
                    size={84}
                    label="What the network sees: your drawing as a 28 by 28 image"
                  />
                )}
              </div>
            </Panel>
          </div>

          <Panel
            title="Prediction"
            info="The network's output: one probability per digit, summing to 100%. It updates live while you train or draw."
          >
            <ProbBars probs={probs} truth={truth} />
          </Panel>

          <Panel
            title={
              cnn ? 'Convolution filters (layer 1)' : 'First-layer weights (16 of 128 neurons)'
            }
            info={
              cnn
                ? 'Each tiny image is one 5×5 filter that slides over the picture looking for a pattern (an edge, a stroke). Red = the filter wants bright pixels there, blue = dark. At the start they are random noise; watch them turn into edge and stroke detectors as training progresses.'
                : 'Each square shows the 784 weights of one hidden neuron laid out as a 28×28 image: red where it likes bright pixels, blue where it likes dark ones. They start as noise and gradually become stroke-shaped templates.'
            }
          >
            {snap ? (
              <MapGrid
                stack={snap.filters}
                cols={cnn ? 8 : 8}
                mode="signed"
                label="Learned filters, red positive and blue negative"
              />
            ) : (
              <p className="text-xs text-muted">Waiting for the model…</p>
            )}
          </Panel>

          {cnn ? (
            <div className="grid gap-3 lg:grid-cols-2">
              <Panel
                title="Feature maps · conv 1"
                info="What the 8 filters of the first layer 'see' in the current image: bright pixels are where that filter found its pattern. Early maps look like edge-highlighted copies of the digit."
              >
                {s.focus?.conv1 ? (
                  <MapGrid
                    stack={s.focus.conv1}
                    cols={4}
                    mode="positive"
                    perTile
                    label="Feature maps of convolution layer 1"
                  />
                ) : (
                  <p className="text-xs text-muted">Pick or draw an image.</p>
                )}
              </Panel>
              <Panel
                title="Feature maps · conv 2"
                info="The 16 maps of the second convolution layer (after pooling). They combine edges from layer 1 into more abstract parts: curves, corners, loops."
              >
                {s.focus?.conv2 ? (
                  <MapGrid
                    stack={s.focus.conv2}
                    cols={4}
                    mode="positive"
                    perTile
                    label="Feature maps of convolution layer 2"
                  />
                ) : (
                  <p className="text-xs text-muted">Pick or draw an image.</p>
                )}
              </Panel>
            </div>
          ) : (
            <Panel
              title="Hidden layer 1 activations"
              info="How strongly each of the 128 neurons in the first hidden layer fires for this image. A healthy layer uses many different neurons for different digits."
            >
              {s.focus?.hidden ? (
                <HiddenStrip values={s.focus.hidden} />
              ) : (
                <p className="text-xs text-muted">Pick or draw an image.</p>
              )}
            </Panel>
          )}
        </main>

        {/* ----- right: curves and errors ----- */}
        <aside
          className="space-y-3 md:col-span-2 xl:col-span-1"
          aria-label="Training curves and errors"
        >
          <Panel
            title="Loss"
            info="How wrong the model is (cross-entropy), lower is better. Teal: the running average on training batches. Amber: loss on the 2,000 test images, measured every few frames."
          >
            <LineChart
              label="Training and test loss"
              yMin={0}
              series={[
                { label: 'train', color: '#2dd4bf', x: steps, y: hist.map((p) => p.trainLoss) },
                {
                  label: 'test',
                  color: '#f5b942',
                  x: evals.map((p) => p.step),
                  y: evals.map((p) => p.testLoss as number),
                },
              ]}
            />
          </Panel>
          <Panel
            title="Accuracy over time"
            info="The same story as accuracy percentages. If training accuracy keeps rising but test accuracy flattens, the model is starting to memorise."
          >
            <LineChart
              label="Training and test accuracy"
              yMin={0}
              yMax={1}
              height={120}
              yFormat={(v) => `${Math.round(v * 100)}%`}
              series={[
                { label: 'train', color: '#2dd4bf', x: steps, y: hist.map((p) => p.trainAcc) },
                {
                  label: 'test',
                  color: '#f5b942',
                  x: evals.map((p) => p.step),
                  y: evals.map((p) => p.testAcc as number),
                },
              ]}
            />
          </Panel>
          <Panel
            title="Confusion matrix"
            info="Rows are the true digit, columns the digit the model guessed, on 2,000 test images. A bright diagonal means correct. Bright cells off the diagonal show which digits get mixed up (4↔9 and 3↔5 are classics)."
          >
            {s.lastEval ? (
              <ConfusionMatrix confusion={s.lastEval.confusion} />
            ) : (
              <p className="text-xs text-muted">Appears after the first evaluation.</p>
            )}
          </Panel>
          <Panel
            title="Most confused examples"
            info="The test images the model got wrong while being most confident. Click one to send it through the network and inspect its feature maps. Often these are genuinely ambiguous or sloppily written digits."
          >
            {s.lastEval && s.data ? (
              s.lastEval.worst.length ? (
                <div className="grid grid-cols-4 gap-2">
                  {s.lastEval.worst.map((w) => (
                    <button
                      key={w.index}
                      type="button"
                      onClick={() => s.focusTest(w.index)}
                      className="rounded-lg border border-line p-1 text-center hover:border-accent"
                      aria-label={`Test image ${w.index}: true ${w.truth}, predicted ${w.pred} with ${(w.conf * 100).toFixed(0)} percent confidence`}
                    >
                      <DigitCanvas
                        pixels={testPixels(s.data!, w.index)}
                        size={60}
                        label={`True ${w.truth}`}
                      />
                      <div className="num mt-0.5 text-[10px] text-muted">
                        {w.truth} → <span className="text-pos">{w.pred}</span>{' '}
                        {(w.conf * 100).toFixed(0)}%
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-ok">No mistakes on the test images. 🎉</p>
              )
            ) : (
              <p className="text-xs text-muted">Appears after the first evaluation.</p>
            )}
          </Panel>
        </aside>
      </div>
    </div>
  )
}
