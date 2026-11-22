import { scaleLinear, scaleLog } from 'd3-scale'
import { AutoCanvas } from '../components/AutoCanvas'
import { cssVar } from '../lib/colors'

export interface Series {
  label: string
  color: string
  x: number[]
  y: number[]
  dashed?: boolean
}

interface Props {
  series: Series[]
  height?: number
  label: string
  /** log-scale y axis (for gradient magnitudes) */
  log?: boolean
  yMin?: number
  yMax?: number
  /** vertical marker, e.g. the step we are time-travelling to */
  marker?: number | null
  onPickX?: (x: number) => void
  yFormat?: (v: number) => string
}

const M = { l: 42, r: 8, t: 8, b: 20 }

/** A canvas line chart. Cheap enough to redraw every frame with thousands of points. */
export function LineChart({ series, height = 150, label, log, yMin, yMax, marker, onPickX, yFormat }: Props) {
  const xs = series.flatMap((s) => s.x)
  const xMax = xs.length ? Math.max(...xs) : 1
  const xMin = xs.length ? Math.min(...xs) : 0

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const text = cssVar('--muted', '#8b98a8')
    const grid = cssVar('--border', '#232c38')
    const ink = cssVar('--text', '#e6edf3')
    const accent = cssVar('--accent', '#2dd4bf')
    const iw = w - M.l - M.r
    const ih = h - M.t - M.b

    const ys = series.flatMap((s) => s.y.filter((v) => Number.isFinite(v) && (!log || v > 0)))
    let lo = yMin ?? (ys.length ? Math.min(...ys) : 0)
    let hi = yMax ?? (ys.length ? Math.max(...ys) : 1)
    if (log) {
      lo = Math.max(lo, 1e-12)
      if (hi <= lo) hi = lo * 10
    } else if (hi - lo < 1e-9) {
      hi = lo + 1
    }
    const sx = scaleLinear().domain([xMin, Math.max(xMax, xMin + 1)]).range([M.l, M.l + iw])
    const sy = log
      ? scaleLog().domain([lo, hi]).range([M.t + ih, M.t]).clamp(true)
      : scaleLinear().domain([lo, hi]).range([M.t + ih, M.t]).nice().clamp(true)

    ctx.font = '10px ui-monospace, Consolas, monospace'
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'right'
    // y grid + labels
    const ticks = log ? sy.ticks(4) : sy.ticks(4)
    for (const t of ticks) {
      const y = sy(t)
      ctx.strokeStyle = grid
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(M.l, y)
      ctx.lineTo(M.l + iw, y)
      ctx.stroke()
      ctx.fillStyle = text
      ctx.fillText(yFormat ? yFormat(t) : log ? t.toExponential(0) : Number(t.toPrecision(2)).toString(), M.l - 5, y)
    }
    // x labels
    ctx.textAlign = 'center'
    ctx.textBaseline = 'top'
    for (const t of sx.ticks(5)) {
      ctx.fillStyle = text
      ctx.fillText(String(t), sx(t), h - M.b + 5)
    }

    // series
    ctx.save()
    ctx.beginPath()
    ctx.rect(M.l, M.t - 2, iw, ih + 4)
    ctx.clip()
    for (const s of series) {
      ctx.strokeStyle = s.color
      ctx.lineWidth = 1.6
      ctx.setLineDash(s.dashed ? [4, 3] : [])
      ctx.beginPath()
      let started = false
      for (let i = 0; i < s.x.length; i++) {
        const v = s.y[i]
        if (!Number.isFinite(v) || (log && v <= 0)) continue
        const px = sx(s.x[i])
        const py = sy(v)
        if (!started) {
          ctx.moveTo(px, py)
          started = true
        } else ctx.lineTo(px, py)
      }
      ctx.stroke()
    }
    ctx.restore()
    ctx.setLineDash([])

    // marker for time travel
    if (marker !== null && marker !== undefined) {
      const mx = sx(marker)
      if (mx >= M.l && mx <= M.l + iw) {
        ctx.strokeStyle = accent
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(mx, M.t)
        ctx.lineTo(mx, M.t + ih)
        ctx.stroke()
      }
    }

    // legend
    ctx.textAlign = 'left'
    ctx.textBaseline = 'middle'
    let lx = M.l + 6
    for (const s of series) {
      ctx.fillStyle = s.color
      ctx.fillRect(lx, M.t + 6, 10, 3)
      ctx.fillStyle = ink
      ctx.fillText(s.label, lx + 14, M.t + 8)
      lx += 22 + ctx.measureText(s.label).width
    }
  }

  return (
    <AutoCanvas
      draw={draw}
      height={height}
      label={label}
      className={onPickX ? 'cursor-crosshair' : ''}
      onClick={
        onPickX
          ? (e, w) => {
              const rect = e.currentTarget.getBoundingClientRect()
              const px = e.clientX - rect.left
              const iw = w - M.l - M.r
              const frac = (px - M.l) / iw
              if (frac < 0 || frac > 1) return
              onPickX(xMin + frac * Math.max(xMax - xMin, 1))
            }
          : undefined
      }
    />
  )
}
