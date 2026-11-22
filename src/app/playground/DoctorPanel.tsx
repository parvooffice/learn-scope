import { useRun } from '../../store/run'
import { useUI } from '../../store/ui'
import { Badge, Btn, Panel } from '../../components/ui'
import type { Diagnosis } from '../../doctor/types'

const TONE = { critical: 'pos', warn: 'warn', info: 'ok' } as const
const ICON = { critical: '⛔', warn: '⚠️', info: '✅' } as const

function Card({ d }: { d: Diagnosis }) {
  const jump = useRun((s) => s.jumpToStep)
  const apply = useRun((s) => s.applyFix)
  const flash = useUI((s) => s.flash)
  return (
    <article
      className={`rounded-lg border bg-panel2 p-2.5 ${
        d.severity === 'critical'
          ? 'border-pos/50'
          : d.severity === 'warn'
            ? 'border-warn/50'
            : 'border-line'
      }`}
      aria-label={`${d.severity} diagnosis: ${d.title}`}
    >
      <div className="mb-1 flex items-center justify-between gap-2">
        <h4 className="text-sm font-semibold text-ink">
          <span aria-hidden>{ICON[d.severity]} </span>
          {d.title}
        </h4>
        <Badge tone={TONE[d.severity]}>{d.severity}</Badge>
      </div>
      <p className="text-xs leading-relaxed text-muted">{d.explanation}</p>
      <p className="mt-1.5 text-[11px] text-muted">
        Evidence: <span className="num text-ink">step {d.evidence.step}</span> — {d.evidence.detail}
      </p>
      <div className="mt-2 flex flex-wrap gap-2">
        <Btn
          onClick={() => {
            jump(d.evidence.step)
            flash(d.evidence.chart)
          }}
          className="!py-1 text-xs"
        >
          🔎 Show me
        </Btn>
        {d.fix && (
          <Btn variant="primary" onClick={() => apply(d.fix!)} className="!py-1 text-xs">
            ✨ {d.fix.label}
          </Btn>
        )}
      </div>
    </article>
  )
}

export function DoctorPanel() {
  const diagnoses = useRun((s) => s.diagnoses)
  return (
    <Panel
      title="Model Doctor"
      info="The Doctor reads the telemetry of your training run (gradient sizes, dead neurons, loss curves) and applies a set of deterministic rules to spot common problems. “Show me” jumps back in time to where the problem is visible; the fix button applies a suggested change."
    >
      <div aria-live="polite" className="space-y-2">
        {diagnoses.length === 0 ? (
          <p className="text-xs text-muted">
            Watching the run… diagnoses appear after a few dozen steps.
          </p>
        ) : (
          diagnoses.map((d) => <Card key={d.id} d={d} />)
        )}
      </div>
    </Panel>
  )
}
