import { AutoCanvas } from '../components/AutoCanvas'
import { cssVar, rgb, viridis } from '../lib/colors'
import type { Dataset } from '../engine/types'

export const GRID = 48
export const EXTENT = 1.3

let heat: HTMLCanvasElement | null = null

/** Paint a G×G scalar field (values 0..1) as a smooth heat map. */
export function paintField(
  ctx: CanvasRenderingContext2D,
  values: ArrayLike<number>,
  G: number,
  x: number,
  y: number,
  w: number,
  h: number,
  color: (t: number) => [number, number, number] = viridis,
) {
  if (!heat) heat = document.createElement('canvas')
  heat.width = G
  heat.height = G
  const hctx = heat.getContext('2d')!
  const img = hctx.createImageData(G, G)
  for (let i = 0; i < G * G; i++) {
    const [r, g, b] = color(values[i])
    img.data[i * 4] = r
    img.data[i * 4 + 1] = g
    img.data[i * 4 + 2] = b
    img.data[i * 4 + 3] = 255
  }
  hctx.putImageData(img, 0, 0)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(heat, x, y, w, h)
}

interface Props {
  /** probability of class 1 on the grid (GRID*GRID), or undefined before the first snapshot */
  field?: Float32Array
  train: Dataset
  test: Dataset
  size?: number
  showTest?: boolean
}

export function DecisionBoundary({ field, train, test, size = 340, showTest = true }: Props) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const s = Math.min(w, h)
    const ox = (w - s) / 2
    if (field) paintField(ctx, field, GRID, ox, 0, s, s)
    else {
      ctx.fillStyle = cssVar('--panel2', '#141a22')
      ctx.fillRect(ox, 0, s, s)
    }
    const px = (v: number) => ox + ((v + EXTENT) / (2 * EXTENT)) * s
    const py = (v: number) => ((EXTENT - v) / (2 * EXTENT)) * s
    const point = (d: Dataset, i: number, hollow: boolean) => {
      const cls = d.y[i]
      const cx = px(d.x[i * 2])
      const cy = py(d.x[i * 2 + 1])
      ctx.beginPath()
      ctx.arc(cx, cy, hollow ? 3.6 : 3, 0, Math.PI * 2)
      // class 1 = black dot, class 0 = white dot: both stand out on the viridis background
      const fill = cls === 1 ? '#05070a' : '#ffffff'
      const edge = cls === 1 ? '#ffffff' : '#05070a'
      if (hollow) {
        ctx.lineWidth = 2
        ctx.strokeStyle = fill === '#ffffff' ? '#ffffff' : '#05070a'
        ctx.stroke()
        ctx.lineWidth = 0.8
        ctx.strokeStyle = edge
        ctx.beginPath()
        ctx.arc(cx, cy, 4.6, 0, Math.PI * 2)
        ctx.stroke()
      } else {
        ctx.fillStyle = fill
        ctx.fill()
        ctx.lineWidth = 1
        ctx.strokeStyle = edge
        ctx.stroke()
      }
    }
    for (let i = 0; i < train.n; i++) point(train, i, false)
    if (showTest) for (let i = 0; i < test.n; i++) point(test, i, true)
    ctx.strokeStyle = cssVar('--border', '#232c38')
    ctx.strokeRect(ox + 0.5, 0.5, s - 1, s - 1)
    // legend gradient
    ctx.fillStyle = rgb(viridis(0))
    ctx.fillRect(ox + 6, s - 16, 10, 10)
    ctx.fillStyle = rgb(viridis(1))
    ctx.fillRect(ox + 62, s - 16, 10, 10)
    ctx.fillStyle = '#fff'
    ctx.font = '10px ui-monospace, Consolas, monospace'
    ctx.textBaseline = 'middle'
    ctx.fillText('p=0', ox + 19, s - 11)
    ctx.fillText('p=1', ox + 75, s - 11)
  }
  return (
    <AutoCanvas
      draw={draw}
      height={size}
      label="Decision boundary: the colour shows the network's predicted probability of class 1 at every point; dots are training data, rings are test data."
    />
  )
}
