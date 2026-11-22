import { AutoCanvas } from '../components/AutoCanvas'
import { cssVar } from '../lib/colors'
import type { Hist } from '../engine/types'

interface Props {
  hist: Hist
  color: string
  height?: number
  label: string
  /** draw a faint line at zero */
  zeroLine?: boolean
}

export function Histogram({ hist, color, height = 54, label, zeroLine = true }: Props) {
  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const grid = cssVar('--border', '#232c38')
    const muted = cssVar('--muted', '#8b98a8')
    const max = Math.max(1, ...hist.counts)
    const n = hist.counts.length
    const bw = w / n
    ctx.fillStyle = color
    for (let i = 0; i < n; i++) {
      const bh = (hist.counts[i] / max) * (h - 14)
      ctx.globalAlpha = 0.9
      ctx.fillRect(i * bw + 0.5, h - 12 - bh, Math.max(1, bw - 1), bh)
    }
    ctx.globalAlpha = 1
    ctx.strokeStyle = grid
    ctx.beginPath()
    ctx.moveTo(0, h - 11.5)
    ctx.lineTo(w, h - 11.5)
    ctx.stroke()
    if (zeroLine && hist.lo < 0 && hist.hi > 0) {
      const zx = ((0 - hist.lo) / (hist.hi - hist.lo)) * w
      ctx.strokeStyle = muted
      ctx.setLineDash([2, 2])
      ctx.beginPath()
      ctx.moveTo(zx, 0)
      ctx.lineTo(zx, h - 11)
      ctx.stroke()
      ctx.setLineDash([])
    }
    ctx.fillStyle = muted
    ctx.font = '9px ui-monospace, Consolas, monospace'
    ctx.textBaseline = 'bottom'
    ctx.textAlign = 'left'
    ctx.fillText(Number(hist.lo.toPrecision(2)).toString(), 1, h)
    ctx.textAlign = 'right'
    ctx.fillText(Number(hist.hi.toPrecision(2)).toString(), w - 1, h)
  }
  return <AutoCanvas draw={draw} height={height} label={label} />
}
