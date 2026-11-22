import { useEffect, useRef } from 'react'
import { canvasToMnist } from '../engine/datasets/mnist'
import { Btn } from '../components/ui'

const SIZE = 224

/** Draw a digit with mouse/touch/pen. Every stroke update is sent (throttled) to the network. */
export function DrawPad({ onPixels }: { onPixels: (px: Float32Array) => void }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const drawing = useRef(false)
  const last = useRef<{ x: number; y: number } | null>(null)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const clear = () => {
    const c = ref.current!
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, c.width, c.height)
  }
  useEffect(clear, [])

  const emit = () => {
    clearTimeout(timer.current)
    timer.current = setTimeout(() => onPixels(canvasToMnist(ref.current!)), 60)
  }
  const pos = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect()
    return {
      x: ((e.clientX - r.left) / r.width) * SIZE,
      y: ((e.clientY - r.top) / r.height) * SIZE,
    }
  }
  const stroke = (e: React.PointerEvent) => {
    const ctx = ref.current!.getContext('2d')!
    const p = pos(e)
    ctx.strokeStyle = '#fff'
    ctx.fillStyle = '#fff'
    ctx.lineWidth = 16
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
    ctx.beginPath()
    if (last.current) {
      ctx.moveTo(last.current.x, last.current.y)
      ctx.lineTo(p.x, p.y)
      ctx.stroke()
    } else {
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2)
      ctx.fill()
    }
    last.current = p
    emit()
  }

  return (
    <div>
      <canvas
        ref={ref}
        width={SIZE}
        height={SIZE}
        role="img"
        aria-label="Drawing pad: draw a digit with the mouse, finger or pen and the network predicts it live"
        className="touch-none rounded-lg border border-line"
        style={{ width: SIZE, height: SIZE, cursor: 'crosshair', background: '#000' }}
        onPointerDown={(e) => {
          ;(e.target as HTMLCanvasElement).setPointerCapture(e.pointerId)
          drawing.current = true
          last.current = null
          stroke(e)
        }}
        onPointerMove={(e) => drawing.current && stroke(e)}
        onPointerUp={() => {
          drawing.current = false
          last.current = null
        }}
        onPointerCancel={() => {
          drawing.current = false
          last.current = null
        }}
      />
      <div className="mt-2 flex gap-2">
        <Btn
          onClick={() => {
            clear()
            onPixels(canvasToMnist(ref.current!))
          }}
        >
          Clear
        </Btn>
      </div>
    </div>
  )
}
