import { history, useRun, viewSnapshot } from '../../store/run'
import { Badge, Btn, Info } from '../../components/ui'

export function Timeline() {
  const version = useRun((s) => s.version)
  const selectedStep = useRun((s) => s.selectedStep)
  const scrubTo = useRun((s) => s.scrubTo)
  const nudge = useRun((s) => s.nudge)
  const goLive = useRun((s) => s.goLive)
  const diagnoses = useRun((s) => s.diagnoses)
  void version

  const n = history.length
  const view = viewSnapshot(selectedStep)
  const idx = selectedStep === null ? n - 1 : history.nearest(selectedStep, (s) => s.step)
  const live = selectedStep === null
  const mb = history.totalBytes / (1024 * 1024)

  return (
    <div
      className="border-t border-line bg-panel px-3 py-2"
      role="region"
      aria-label="Training timeline"
    >
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1.5">
          <Btn onClick={() => nudge(-10)} ariaLabel="Back 10 snapshots" className="!px-2 !py-1">
            «
          </Btn>
          <Btn
            onClick={() => nudge(-1)}
            ariaLabel="Previous snapshot"
            title="Left arrow"
            className="!px-2 !py-1"
          >
            ‹
          </Btn>
          <Btn
            onClick={() => nudge(1)}
            ariaLabel="Next snapshot"
            title="Right arrow (while time travelling)"
            className="!px-2 !py-1"
          >
            ›
          </Btn>
          <Btn onClick={() => nudge(10)} ariaLabel="Forward 10 snapshots" className="!px-2 !py-1">
            »
          </Btn>
        </div>
        <div className="relative min-w-[160px] flex-1">
          <input
            type="range"
            min={0}
            max={Math.max(0, n - 1)}
            value={Math.max(0, idx)}
            onChange={(e) => scrubTo(Number(e.target.value))}
            aria-label="Timeline scrubber: drag to view the network at any recorded training step"
            disabled={n < 2}
          />
          {/* evidence markers from the Model Doctor */}
          <div className="pointer-events-none absolute inset-x-0 -top-1 h-1.5">
            {diagnoses
              .filter((d) => d.id !== 'healthy' && n > 1)
              .map((d) => {
                const i = history.nearest(d.evidence.step, (s) => s.step)
                return (
                  <span
                    key={d.id}
                    title={d.title}
                    className={`absolute h-1.5 w-1.5 -translate-x-1/2 rounded-full ${d.severity === 'critical' ? 'bg-pos' : 'bg-warn'}`}
                    style={{ left: `${(i / Math.max(1, n - 1)) * 100}%` }}
                  />
                )
              })}
          </div>
        </div>
        <div className="num flex items-center gap-2 text-xs text-muted">
          <span>
            step <b className="text-ink">{view?.step ?? 0}</b>
          </span>
          <span>
            epoch <b className="text-ink">{(view?.epoch ?? 0).toFixed(1)}</b>
          </span>
          <span className="hidden sm:inline">
            {n} snapshots · {mb.toFixed(1)} MB
          </span>
        </div>
        {live ? (
          <Badge tone="accent">
            <span className="live-dot mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-accent" />
            LIVE
          </Badge>
        ) : (
          <Btn variant="primary" onClick={goLive} className="!py-1" title="L">
            Back to live
          </Btn>
        )}
        <Info
          title="What is the timeline?"
          text="Every few training steps the app records a snapshot: all weights, gradient sizes, activation statistics and metrics. Drag the slider (or press ← →) to rewind: the network graph, decision boundary and histograms all redraw exactly as they were at that moment. Small coloured dots mark where the Model Doctor found problems."
        />
      </div>
    </div>
  )
}
