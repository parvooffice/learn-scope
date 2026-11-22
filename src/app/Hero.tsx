import { useEffect, useRef, useState } from 'react'
import { CpuMLP } from '../engine/cpu/mlp'
import { generateDataset } from '../engine/datasets/toy'
import { viridis } from '../lib/colors'

const G = 44
const EXTENT = 1.2
const SIZE = 420
const CYCLE_STEPS = 900
const STEPS_PER_FRAME = 2

/**
 * Landing-page hero: a spiral being learned, live, on the CPU in plain JS.
 * Loops forever with a fresh random network every ~10 seconds.
 */
export function Hero() {
  const ref = useRef<HTMLCanvasElement>(null)
  const [hud, setHud] = useState({ step: 0, loss: 0 })

  useEffect(() => {
    const canvas = ref.current!
    const ctx = canvas.getContext('2d')!
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    canvas.width = SIZE * dpr
    canvas.height = SIZE * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    const heat = document.createElement('canvas')
    heat.width = G
    heat.height = G
    const hctx = heat.getContext('2d')!
    const img = hctx.createImageData(G, G)

    let seed = 11
    let net = new CpuMLP([2, 10, 10, 1], seed)
    let ds = generateDataset('spiral', 150, 0.05, seed)
    let step = 0
    let loss = 0
    let raf = 0

    const reinit = () => {
      seed++
      net = new CpuMLP([2, 10, 10, 1], seed)
      ds = generateDataset('spiral', 150, 0.05, seed)
      step = 0
    }

    const draw = () => {
      for (let gy = 0; gy < G; gy++) {
        for (let gx = 0; gx < G; gx++) {
          const x0 = -EXTENT + ((gx + 0.5) / G) * 2 * EXTENT
          const x1 = EXTENT - ((gy + 0.5) / G) * 2 * EXTENT
          const [r, g, b] = viridis(net.predict(x0, x1))
          const i = (gy * G + gx) * 4
          img.data[i] = r
          img.data[i + 1] = g
          img.data[i + 2] = b
          img.data[i + 3] = 255
        }
      }
      hctx.putImageData(img, 0, 0)
      ctx.imageSmoothingEnabled = true
      ctx.drawImage(heat, 0, 0, SIZE, SIZE)
      for (let i = 0; i < ds.n; i++) {
        const cx = ((ds.x[i * 2] + EXTENT) / (2 * EXTENT)) * SIZE
        const cy = ((EXTENT - ds.x[i * 2 + 1]) / (2 * EXTENT)) * SIZE
        ctx.beginPath()
        ctx.arc(cx, cy, 3.2, 0, Math.PI * 2)
        ctx.fillStyle = ds.y[i] ? '#05070a' : '#ffffff'
        ctx.fill()
        ctx.lineWidth = 1
        ctx.strokeStyle = ds.y[i] ? '#ffffff' : '#05070a'
        ctx.stroke()
      }
    }

    if (reduce) {
      // reduced motion: train silently and show the finished picture
      for (let i = 0; i < CYCLE_STEPS; i++) loss = net.trainStep(ds.x, ds.y, 0.03)
      step = CYCLE_STEPS
      draw()
      setHud({ step, loss })
      return
    }

    const frame = () => {
      for (let k = 0; k < STEPS_PER_FRAME; k++) {
        loss = net.trainStep(ds.x, ds.y, 0.03)
        step++
      }
      draw()
      if (step % 20 === 0) setHud({ step, loss })
      if (step >= CYCLE_STEPS + 300) reinit() // linger on the finished picture, then start over
      raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <div className="relative mx-auto w-full max-w-[420px] overflow-hidden rounded-2xl border border-line shadow-[0_0_80px_-20px_var(--accent)]">
      <canvas
        ref={ref}
        style={{ width: '100%', aspectRatio: '1 / 1' }}
        role="img"
        aria-label="Animation: a small neural network learning to separate two interleaved spirals. The coloured background is the network's current prediction."
      />
      <div className="num absolute left-2 top-2 rounded-md bg-black/55 px-2 py-1 text-[11px] text-white backdrop-blur">
        step {hud.step} · loss {hud.loss.toFixed(3)}
      </div>
      <div className="absolute bottom-2 right-2 rounded-md bg-black/55 px-2 py-1 text-[11px] text-white backdrop-blur">
        live · running in your browser
      </div>
    </div>
  )
}
