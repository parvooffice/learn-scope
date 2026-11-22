import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { useUI } from '../store/ui'
import type { ChartId } from '../doctor/types'

/** "ⓘ What am I looking at?" — a short plain-English explanation of a chart. */
export function Info({
  text,
  title = 'What am I looking at?',
}: {
  text: ReactNode
  title?: string
}) {
  const [open, setOpen] = useState(false)
  const id = useId()
  const ref = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent) {
        if (e.key === 'Escape') setOpen(false)
      } else if (!ref.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    document.addEventListener('keydown', close)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('keydown', close)
    }
  }, [open])
  return (
    <span ref={ref} className="relative inline-block">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        aria-label={title}
        title={title}
        onClick={() => setOpen((o) => !o)}
        className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-line text-[11px] text-muted hover:border-accent hover:text-accent"
      >
        i
      </button>
      {open && (
        <span
          id={id}
          role="note"
          className="absolute right-0 top-6 z-30 w-64 rounded-lg border border-line bg-panel2 p-3 text-xs font-normal leading-relaxed text-ink shadow-xl"
        >
          <strong className="mb-1 block text-accent">{title}</strong>
          {text}
        </span>
      )}
    </span>
  )
}

export function Panel({
  title,
  info,
  children,
  chart,
  actions,
  className = '',
}: {
  title: string
  info?: ReactNode
  children: ReactNode
  /** if the Model Doctor points at this chart, the panel flashes */
  chart?: ChartId
  actions?: ReactNode
  className?: string
}) {
  const highlight = useUI((s) => s.highlight)
  const [flashing, setFlashing] = useState(false)
  useEffect(() => {
    if (!highlight || highlight.chart !== chart) return
    setFlashing(true)
    const t = setTimeout(() => setFlashing(false), 1800)
    return () => clearTimeout(t)
  }, [highlight, chart])
  return (
    <section
      data-chart={chart}
      className={`rounded-xl border bg-panel transition-colors ${
        flashing ? 'border-accent shadow-[0_0_0_2px_var(--accent)]' : 'border-line'
      } ${className}`}
    >
      <header className="flex items-center justify-between gap-2 px-3 pt-2.5 pb-1.5">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">{title}</h3>
        <div className="flex items-center gap-2">
          {actions}
          {info && <Info text={info} />}
        </div>
      </header>
      <div className="px-3 pb-3">{children}</div>
    </section>
  )
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  format?: (v: number) => string
  hint?: string
}) {
  const id = useId()
  return (
    <div className="space-y-0.5" title={hint}>
      <div className="flex items-baseline justify-between text-xs">
        <label htmlFor={id} className="text-muted">
          {label}
        </label>
        <output htmlFor={id} className="num text-ink">
          {format ? format(value) : value}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

export function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  const id = useId()
  return (
    <div className="space-y-0.5">
      <label htmlFor={id} className="block text-xs text-muted">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="w-full rounded-md border border-line bg-panel2 px-2 py-1.5 text-sm text-ink"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

type BtnProps = {
  children: ReactNode
  onClick?: () => void
  variant?: 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  title?: string
  ariaLabel?: string
  className?: string
}
export function Btn({
  children,
  onClick,
  variant = 'ghost',
  disabled,
  title,
  ariaLabel,
  className = '',
}: BtnProps) {
  const base =
    'inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors select-none'
  const styles = {
    primary: 'bg-accent text-accent-ink hover:brightness-110',
    ghost: 'border border-line bg-panel2 text-ink hover:border-accent',
    danger: 'border border-line bg-panel2 text-pos hover:border-pos',
  }[variant]
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      className={`${base} ${styles} ${className}`}
    >
      {children}
    </button>
  )
}

export function Badge({
  children,
  tone = 'muted',
}: {
  children: ReactNode
  tone?: 'muted' | 'ok' | 'warn' | 'pos' | 'accent'
}) {
  const t = {
    muted: 'border-line text-muted',
    ok: 'border-ok/40 text-ok',
    warn: 'border-warn/50 text-warn',
    pos: 'border-pos/50 text-pos',
    accent: 'border-accent/50 text-accent',
  }[tone]
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] ${t}`}>
      {children}
    </span>
  )
}
