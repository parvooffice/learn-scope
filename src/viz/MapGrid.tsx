import { AutoCanvas } from '../components/AutoCanvas'
import { diverging, viridis } from '../lib/colors'
import type { MapStack } from '../worker/mnistProtocol'

interface Props {
  stack: MapStack
  cols: number
  label: string
  /** 'signed' = blue/red diverging around 0 (filters); 'positive' = viridis (feature maps) */
  mode: 'signed' | 'positive'
  gap?: number
  /** normalise each tile separately (good for filters) or all tiles together */
  perTile?: boolean
}

let tile: HTMLCanvasElement | null = null

/** A grid of small images (conv filters, feature maps). */
export function MapGrid({ stack, cols, label, mode, gap = 4, perTile = true }: Props) {
  const rows = Math.ceil(stack.count / cols)
  const draw = (ctx: CanvasRenderingContext2D, w: number) => {
    const cell = (w - gap * (cols - 1)) / cols
    const n = stack.size
    if (!tile) tile = document.createElement('canvas')
    tile.width = n
    tile.height = n
    const tctx = tile.getContext('2d')!
    const img = tctx.createImageData(n, n)
    let gMax = 1e-9
    let gMin = Infinity
    if (!perTile)
      for (let i = 0; i < stack.data.length; i++) {
        gMax = Math.max(gMax, Math.abs(stack.data[i]))
        gMin = Math.min(gMin, stack.data[i])
      }
    for (let k = 0; k < stack.count; k++) {
      const off = k * n * n
      let mx = 1e-9
      let mn = Infinity
      if (perTile) {
        for (let i = 0; i < n * n; i++) {
          mx = Math.max(mx, Math.abs(stack.data[off + i]))
          mn = Math.min(mn, stack.data[off + i])
        }
      } else {
        mx = gMax
        mn = gMin
      }
      for (let i = 0; i < n * n; i++) {
        const v = stack.data[off + i]
        const [r, g, b] =
          mode === 'signed'
            ? diverging(v / mx, [24, 28, 36])
            : viridis(
                mx - Math.max(0, mn) > 1e-9 ? (v - Math.max(0, mn)) / (mx - Math.max(0, mn)) : 0,
              )
        img.data[i * 4] = r
        img.data[i * 4 + 1] = g
        img.data[i * 4 + 2] = b
        img.data[i * 4 + 3] = 255
      }
      tctx.putImageData(img, 0, 0)
      ctx.imageSmoothingEnabled = n > 12
      const cx = (k % cols) * (cell + gap)
      const cy = Math.floor(k / cols) * (cell + gap)
      ctx.drawImage(tile, cx, cy, cell, cell)
    }
  }
  return (
    <AutoCanvas
      draw={draw}
      height={(w) => rows * ((w - gap * (cols - 1)) / cols + gap) - gap}
      label={label}
    />
  )
}
