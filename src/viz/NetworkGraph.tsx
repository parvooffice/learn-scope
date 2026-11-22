import { useRef, useState } from 'react'
import { AutoCanvas } from '../components/AutoCanvas'
import { cssVar, diverging, rgb, viridis } from '../lib/colors'
import type { PlaygroundConfig, Snapshot } from '../engine/types'
import { layerSpecs } from '../engine/types'

export interface NeuronRef {
  /** index into layerSpecs (0 = first hidden layer) */
  layer: number
  unit: number
}

interface Props {
  config: PlaygroundConfig
  snap?: Snapshot
  selected: NeuronRef | null
  onSelect: (n: NeuronRef | null) => void
  height?: number
}

interface NodePos {
  x: number
  y: number
  r: number
  layer: number // -1 = input
  unit: number
}

export function NetworkGraph({ config, snap, selected, onSelect, height = 360 }: Props) {
  const nodes = useRef<NodePos[]>([])
  const [hover, setHover] = useState<{ n: NodePos; x: number; y: number } | null>(null)
  const specs = layerSpecs(config)
  const cols = [2, ...specs.map((s) => s.nout)]

  const draw = (ctx: CanvasRenderingContext2D, w: number, h: number) => {
    const padX = 38
    const padY = 22
    const maxUnits = Math.max(...cols)
    const r = Math.max(5, Math.min(15, (h - 2 * padY) / (maxUnits * 2.5)))
    const pos: NodePos[][] = cols.map((n, c) => {
      const x = cols.length === 1 ? w / 2 : padX + (c * (w - 2 * padX)) / (cols.length - 1)
      return Array.from({ length: n }, (_, u) => ({
        x,
        y: n === 1 ? h / 2 : padY + ((u + 0.5) * (h - 2 * padY)) / n,
        r,
        layer: c - 1,
        unit: u,
      }))
    })
    nodes.current = pos.flat()

    const text = cssVar('--muted', '#8b98a8')
    const ink = cssVar('--text', '#e6edf3')
    const accent = cssVar('--accent', '#2dd4bf')
    const bg = cssVar('--canvas-bg', '#0a0e14')

    // edges
    if (snap) {
      let maxAbs = 1e-9
      for (const wt of snap.weights)
        for (let i = 0; i < wt.W.length; i++) maxAbs = Math.max(maxAbs, Math.abs(wt.W[i]))
      type E = { a: NodePos; b: NodePos; w: number }
      const edges: E[] = []
      snap.weights.forEach((wt, l) => {
        const { nin, nout } = specs[l]
        for (let i = 0; i < nin; i++)
          for (let j = 0; j < nout; j++)
            edges.push({ a: pos[l][i], b: pos[l + 1][j], w: wt.W[i * nout + j] })
      })
      edges.sort((p, q) => Math.abs(p.w) - Math.abs(q.w))
      for (const e of edges) {
        const m = Math.abs(e.w) / maxAbs
        ctx.strokeStyle = rgb(
          diverging(Math.sign(e.w) * 1, [120, 130, 145]),
          0.12 + 0.75 * Math.sqrt(m),
        )
        ctx.lineWidth = 0.3 + 4.2 * m
        ctx.beginPath()
        ctx.moveTo(e.a.x + r, e.a.y)
        ctx.lineTo(e.b.x - r, e.b.y)
        ctx.stroke()
      }
    }

    // nodes
    ctx.font = '10px ui-monospace, Consolas, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    for (const col of pos) {
      for (const n of col) {
        let fill = '#6b7686'
        let dead = false
        if (n.layer >= 0 && snap) {
          const st = snap.layers[n.layer]
          const m = st.unitMean[n.unit] ?? 0
          const act = specs[n.layer].activation
          const t =
            act === 'tanh' ? (m + 1) / 2 : act === 'sigmoid' ? m : m / Math.max(0.5, ...st.unitMean)
          fill = rgb(viridis(t))
          dead = (act === 'relu' || act === 'gelu') && m < 1e-3 && st.deadPct > 0
        }
        ctx.beginPath()
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        ctx.fillStyle = dead ? bg : fill
        ctx.fill()
        const isSel = selected && selected.layer === n.layer && selected.unit === n.unit
        ctx.lineWidth = isSel ? 3 : 1.2
        ctx.strokeStyle = isSel ? accent : dead ? text : cssVar('--border', '#232c38')
        if (dead) ctx.setLineDash([2, 2])
        ctx.stroke()
        ctx.setLineDash([])
        if (dead) {
          ctx.fillStyle = text
          ctx.fillText('×', n.x, n.y + 0.5)
        }
      }
    }

    // column captions
    ctx.fillStyle = text
    ctx.textBaseline = 'alphabetic'
    pos.forEach((col, c) => {
      const label = c === 0 ? 'input' : c === pos.length - 1 ? 'output' : `hidden ${c}`
      ctx.fillText(label, col[0].x, 12)
    })
    ctx.fillStyle = ink
    ctx.textBaseline = 'middle'
    pos[0].forEach((n, i) => ctx.fillText(i === 0 ? 'x' : 'y', n.x, n.y))
    const o = pos[pos.length - 1][0]
    ctx.fillText('ŷ', o.x, o.y)
  }

  const find = (x: number, y: number): NodePos | null => {
    let best: NodePos | null = null
    let bd = Infinity
    for (const n of nodes.current) {
      const d = Math.hypot(n.x - x, n.y - y)
      if (d <= n.r + 4 && d < bd) {
        best = n
        bd = d
      }
    }
    return best
  }

  return (
    <div className="relative">
      <AutoCanvas
        draw={draw}
        height={height}
        label="Network graph: neurons as circles, weights as lines. Blue lines are negative weights, red are positive, thicker means larger. Click a hidden neuron to see what it detects."
        className="cursor-pointer"
        onClick={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          const n = find(e.clientX - rect.left, e.clientY - rect.top)
          if (!n || n.layer < 0) return onSelect(null)
          onSelect({ layer: n.layer, unit: n.unit })
        }}
        onMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect()
          const x = e.clientX - rect.left
          const y = e.clientY - rect.top
          const n = find(x, y)
          setHover(n ? { n, x, y } : null)
        }}
        onLeave={() => setHover(null)}
      />
      {hover && snap && hover.n.layer >= 0 && (
        <div
          className="pointer-events-none absolute z-10 rounded-md border border-line bg-panel2 px-2 py-1 text-[11px] text-ink shadow"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          layer {hover.n.layer + 1} · unit {hover.n.unit + 1}
          <br />
          mean activation{' '}
          <span className="num">
            {(snap.layers[hover.n.layer].unitMean[hover.n.unit] ?? 0).toFixed(3)}
          </span>
        </div>
      )}
    </div>
  )
}
