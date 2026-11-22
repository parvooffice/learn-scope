import { useEffect, useRef, useState } from 'react'
import { history } from '../../store/run'
import { Btn, Panel } from '../../components/ui'
import { AutoCanvas } from '../../components/AutoCanvas'
import { paintField } from '../../viz/DecisionBoundary'
import { computeLandscape, flatten, project, type Landscape } from '../../engine/landscape'
import { snapshotMatches } from '../../store/view'
import { cssVar } from '../../lib/colors'
import type { Dataset, LayerSpec, Snapshot } from '../../engine/types'

interface Props {
  specs: LayerSpec[]
  train: Dataset
  view?: Snapshot
}

const G = 31
const RANGE = 1

/** A 2-D slice of the loss surface around the weights at the selected step, with the training path. */
export function LossLandscape({ specs, train, view }: Props) {
  const [result, setResult] = useState<{ land: Landscape; step: number } | null>(null)
  const [progress, setProgress] = useState<number | null>(null)
  const cancel = useRef({ cancelled: false })

  // a different architecture/dataset makes the old slice meaningless
  useEffect(() => {
    cancel.current.cancelled = true
    setResult(null)
    setProgress(null)
  }, [specs, train])
  useEffect(() => () => void (cancel.current.cancelled = true), [])

  const compute = async () => {
    if (!view) return
    cancel.current.cancelled = true
    const signal = { cancelled: false }
    cancel.current = signal
    setProgress(0)
    const land = await computeLandscape(specs, view.weights, train.x, train.y, {
      G,
      range: RANGE,
      seed: 1,
      onProgress: setProgress,
      signal,
    })
    if (land && !signal.cancelled) setResult({ land, step: view.step })
    setProgress(null)
  }

  // training path projected into the slice (every snapshot up to the selected step)
  const path: { a: number; b: number; step: number }[] = []
  if (result && view) {
    const center = flatten(view.weights)
    const snaps = history.toArray()
    const stride = Math.max(1, Math.floor(snaps.length / 300))
    for (let i = 0; i < snaps.length; i += stride) {
      const s = snaps[i]
      if (s.step > result.step || !snapshotMatches(specs, s)) continue
      const p = project(flatten(s.weights), center, result.land.d1, result.land.d2)
      path.push({ ...p, step: s.step })
    }
  }

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const s = Math.min(w, h)
    ctx.translate((w - s) / 2, 0)
    if (!result) {
      ctx.fillStyle = cssVar('--panel2', '#141a22')
      ctx.fillRect(0, 0, s, s)
      return
    }
    const { land } = result
    const span = Math.max(land.clip - land.min, 1e-9)
    const t = new Float32Array(G * G)
    for (let i = 0; i < t.length; i++) t[i] = Math.min(1, (land.values[i] - land.min) / span)
    paintField(ctx, t, G, 0, 0, s, s)
    const px = (a: number) => ((a + RANGE) / (2 * RANGE)) * s
    const py = (b: number) => ((RANGE - b) / (2 * RANGE)) * s
    ctx.save()
    ctx.beginPath()
    ctx.rect(0, 0, s, s)
    ctx.clip()
    ctx.strokeStyle = '#ffffff'
    ctx.lineWidth = 1.6
    ctx.globalAlpha = 0.9
    ctx.beginPath()
    path.forEach((p, i) => (i ? ctx.lineTo(px(p.a), py(p.b)) : ctx.moveTo(px(p.a), py(p.b))))
    ctx.stroke()
    ctx.globalAlpha = 1
    if (path.length) {
      ctx.fillStyle = '#05070a'
      ctx.strokeStyle = '#fff'
      ctx.beginPath()
      ctx.arc(px(path[0].a), py(path[0].b), 3.5, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }
    // current weights are the centre of the slice
    ctx.fillStyle = cssVar('--accent', '#2dd4bf')
    ctx.strokeStyle = '#05070a'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    ctx.arc(px(0), py(0), 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
    ctx.restore()
    ctx.strokeStyle = cssVar('--border', '#232c38')
    ctx.strokeRect(0.5, 0.5, s - 1, s - 1)
  }

  return (
    <Panel
      title="Loss landscape"
      info="The loss depends on every weight, far too many dimensions to draw. This picks two random directions in weight space and shows the loss on the flat slice through your current weights (teal dot). Dark = low loss (a valley), bright = high. The white line is where training came from, projected onto the slice; it can leave the picture because the real path uses many more directions. Compute it at different steps with the timeline."
      actions={
        <Btn
          onClick={compute}
          disabled={!view || progress !== null}
          className="!px-2 !py-0.5 text-xs"
        >
          {progress !== null
            ? `${Math.round(progress * 100)}%`
            : result
              ? 'Recompute here'
              : 'Compute'}
        </Btn>
      }
    >
      <AutoCanvas
        height={(w) => Math.min(w, 320)}
        label="Loss landscape: a 2-D slice of the loss surface around the current weights, dark is low loss, with the training path drawn as a white line"
        draw={draw}
      />
      <p className="mt-1 text-[11px] text-muted">
        {result ? (
          <>
            slice at step <span className="num text-ink">{result.step}</span> · loss at centre{' '}
            <span className="num text-ink">{result.land.centerLoss.toFixed(3)}</span> · colour range{' '}
            <span className="num">
              {result.land.min.toFixed(2)}–{result.land.clip.toFixed(2)}
            </span>
          </>
        ) : (
          'Press Compute (about a second) to slice the loss surface at the step you are viewing.'
        )}
      </p>
    </Panel>
  )
}
