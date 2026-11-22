import { useEffect, useRef, type CSSProperties, type MouseEvent } from 'react'
import { useUI } from '../store/ui'

interface Props {
  /** Draw in CSS pixels; the context is already scaled for the device pixel ratio. */
  draw: (ctx: CanvasRenderingContext2D, width: number, height: number) => void
  /** fixed CSS pixels, or a function of the measured width */
  height: number | ((width: number) => number)
  label: string
  className?: string
  style?: CSSProperties
  onClick?: (e: MouseEvent<HTMLCanvasElement>, w: number, h: number) => void
  onMove?: (e: MouseEvent<HTMLCanvasElement>, w: number, h: number) => void
  onLeave?: () => void
  /** width is taken from the parent; pass a number for a fixed square-ish canvas */
  fixedWidth?: number
}

/** A canvas that fills its parent's width, stays crisp on retina screens, and redraws on resize/theme change. */
export function AutoCanvas({
  draw,
  height,
  label,
  className,
  style,
  onClick,
  onMove,
  onLeave,
  fixedWidth,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null)
  const wrap = useRef<HTMLDivElement>(null)
  const drawRef = useRef(draw)
  const theme = useUI((s) => s.theme)
  drawRef.current = draw

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const render = () => {
      const w = fixedWidth ?? Math.max(10, Math.floor(wrap.current?.clientWidth ?? 300))
      const h = typeof height === 'function' ? Math.max(1, Math.round(height(w))) : height
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(w * dpr)
        canvas.height = Math.floor(h * dpr)
        canvas.style.width = `${w}px`
        canvas.style.height = `${h}px`
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      drawRef.current(ctx, w, h)
    }
    render()
    const ro = new ResizeObserver(render)
    if (wrap.current) ro.observe(wrap.current)
    ;(canvas as HTMLCanvasElement & { __render?: () => void }).__render = render
    return () => ro.disconnect()
  }, [height, fixedWidth, theme])

  // redraw after every render (the parent re-renders when its data changes)
  useEffect(() => {
    const canvas = ref.current as (HTMLCanvasElement & { __render?: () => void }) | null
    canvas?.__render?.()
  })

  const dims = (e: MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect()
    return { x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height }
  }
  return (
    <div ref={wrap} className="w-full" style={{ lineHeight: 0 }}>
      <canvas
        ref={ref}
        role="img"
        aria-label={label}
        className={className}
        style={style}
        onClick={onClick ? (e) => onClick(e, dims(e).w, dims(e).h) : undefined}
        onMouseMove={onMove ? (e) => onMove(e, dims(e).w, dims(e).h) : undefined}
        onMouseLeave={onLeave}
      />
    </div>
  )
}
