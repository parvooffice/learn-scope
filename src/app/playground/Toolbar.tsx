import { useRun } from '../../store/run'
import { Badge, Btn } from '../../components/ui'
import { encodeConfig } from '../../lib/urlConfig'
import { useState } from 'react'

const SPEEDS = [
  { label: '1×', v: 1 },
  { label: '4×', v: 4 },
  { label: '16×', v: 16 },
  { label: '64×', v: 64 },
]

export function Toolbar() {
  const running = useRun((s) => s.running)
  const ready = useRun((s) => s.ready)
  const speed = useRun((s) => s.speed)
  const backend = useRun((s) => s.backend)
  const toggle = useRun((s) => s.toggle)
  const step = useRun((s) => s.step)
  const reset = useRun((s) => s.reset)
  const setSpeed = useRun((s) => s.setSpeed)
  const config = useRun((s) => s.config)
  const [copied, setCopied] = useState(false)

  const share = async () => {
    const url = `${window.location.origin}${window.location.pathname}#/playground?c=${encodeConfig(config)}`
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      window.prompt('Copy this link:', url)
    }
  }

  return (
    <div
      className="flex flex-wrap items-center gap-2"
      role="toolbar"
      aria-label="Training controls"
    >
      <Btn
        variant="primary"
        onClick={toggle}
        disabled={!ready}
        ariaLabel={running ? 'Pause training' : 'Start training'}
        title="Space"
      >
        {running ? '⏸ Pause' : '▶ Play'}
      </Btn>
      <Btn onClick={() => step(1)} disabled={!ready} title="Right arrow">
        Step
      </Btn>
      <Btn onClick={() => step(10)} disabled={!ready} title="Shift + right arrow">
        Step ×10
      </Btn>
      <Btn onClick={reset} disabled={!ready} title="R">
        ↺ Reset
      </Btn>
      <div
        className="ml-1 flex items-center gap-1"
        role="group"
        aria-label="Training speed (steps per frame)"
      >
        <span className="text-xs text-muted">speed</span>
        {SPEEDS.map((s) => (
          <button
            key={s.v}
            type="button"
            aria-pressed={speed === s.v}
            onClick={() => setSpeed(s.v)}
            className={`rounded-md border px-2 py-1 text-xs ${
              speed === s.v ? 'border-accent text-accent' : 'border-line text-muted hover:text-ink'
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="ml-auto flex items-center gap-2">
        {backend && (
          <Badge tone={backend === 'webgpu' ? 'accent' : backend === 'webgl' ? 'ok' : 'warn'}>
            {backend === 'webgpu'
              ? 'GPU · WebGPU'
              : backend === 'webgl'
                ? 'GPU · WebGL'
                : 'CPU fallback'}
          </Badge>
        )}
        <Btn onClick={share} title="Copy a link that reopens this exact experiment">
          {copied ? '✓ Link copied' : '🔗 Share'}
        </Btn>
      </div>
    </div>
  )
}
