import { create } from 'zustand'
import {
  DEFAULT_CONFIG,
  snapshotBytes,
  type PlaygroundConfig,
  type Snapshot,
} from '../engine/types'
import { HistoryBuffer } from '../lib/ringBuffer'
import { diagnose } from '../doctor/doctor'
import type { Diagnosis, Fix } from '../doctor/types'
import { send as defaultSend, onMessage as defaultOnMessage } from '../engine/client'
import type { FromWorker, ToWorker } from '../worker/protocol'
import { useUI } from './ui'

const LIVE_KEYS = new Set<keyof PlaygroundConfig>(['lr', 'l2', 'batchSize'])

/** How a run talks to its training worker (the playground uses a shared one; Compare makes two). */
export interface RunClient {
  send: (msg: ToWorker) => void
  onMessage: (fn: (m: FromWorker) => void) => unknown
}

export interface RunState {
  config: PlaygroundConfig
  ready: boolean
  backend: string | null
  notes: string[]
  error: string | null
  running: boolean
  speed: number
  /** bumps whenever `history` changes */
  version: number
  /** null = follow the live head; otherwise a training step we are time-travelling to */
  selectedStep: number | null
  diagnoses: Diagnosis[]

  boot: () => void
  setConfig: (patch: Partial<PlaygroundConfig>) => void
  replaceConfig: (cfg: PlaygroundConfig) => void
  play: () => void
  pause: () => void
  toggle: () => void
  step: (n: number) => void
  reset: () => void
  setSpeed: (s: number) => void
  scrubTo: (index: number) => void
  jumpToStep: (step: number) => void
  goLive: () => void
  nudge: (delta: number) => void
  applyFix: (fix: Fix) => void
}

/**
 * Everything about one training run: config, worker connection, recorded history, diagnoses.
 * Each call creates an independent run (own history buffer, own timers).
 */
export function createRunStore(client: RunClient, initial: Partial<PlaygroundConfig> = {}) {
  const { send, onMessage } = client
  /** Every recorded snapshot. Lives outside React state: it is big and changes constantly. */
  const history = new HistoryBuffer<Snapshot>(snapshotBytes, 3000, 150 * 1024 * 1024)
  let booted = false
  let wantRunAfterBuild = false
  let configureTimer: ReturnType<typeof setTimeout> | undefined
  let doctorTimer: ReturnType<typeof setTimeout> | undefined
  let pending: Snapshot[] = []
  let raf = 0

  const useRun = create<RunState>((set, get) => {
    // THROTTLE, not debounce: snapshots can arrive every frame, and a debounce that restarts its
    // timer on each one would never fire. One run is scheduled at a time; it reads the newest history.
    const runDoctor = () => {
      if (doctorTimer !== undefined) return
      doctorTimer = setTimeout(() => {
        doctorTimer = undefined
        const { config } = get()
        set({ diagnoses: diagnose({ history: history.toArray(), config }) })
      }, 250)
    }

    const flush = () => {
      raf = 0
      if (pending.length === 0) return
      for (const s of pending) history.push(s)
      pending = []
      set((st) => ({ version: st.version + 1 }))
      runDoctor()
    }

    const queueSnapshot = (s: Snapshot) => {
      pending.push(s)
      if (!raf) raf = requestAnimationFrame(flush)
    }

    const scheduleConfigure = (immediate = false) => {
      clearTimeout(configureTimer)
      const go = () => {
        wantRunAfterBuild = get().running || wantRunAfterBuild
        send({ type: 'configure', config: get().config })
      }
      if (immediate) go()
      else configureTimer = setTimeout(go, 200)
    }

    return {
      config: { ...DEFAULT_CONFIG, ...initial },
      ready: false,
      backend: null,
      notes: [],
      error: null,
      running: false,
      speed: 4,
      version: 0,
      selectedStep: null,
      diagnoses: [],

      boot: () => {
        if (booted) return
        booted = true
        onMessage((msg) => {
          switch (msg.type) {
            case 'ready':
              set({ ready: true, backend: msg.backend, notes: msg.notes })
              break
            case 'built':
              pending = []
              history.clear()
              set((st) => ({
                version: st.version + 1,
                selectedStep: null,
                diagnoses: [],
                error: null,
              }))
              if (wantRunAfterBuild) {
                wantRunAfterBuild = false
                send({ type: 'start' })
              }
              break
            case 'snapshot':
              queueSnapshot(msg.snapshot)
              break
            case 'status':
              set({ running: msg.running })
              break
            case 'error':
              set({ error: msg.message, running: false })
              break
          }
        })
        send({ type: 'setSpeed', stepsPerTick: get().speed })
        send({ type: 'configure', config: get().config })
      },

      setConfig: (patch) => {
        const keys = Object.keys(patch) as (keyof PlaygroundConfig)[]
        set({ config: { ...get().config, ...patch } })
        const liveOnly = keys.every((k) => LIVE_KEYS.has(k))
        if (liveOnly) {
          send({ type: 'live', lr: patch.lr, l2: patch.l2, batchSize: patch.batchSize })
        } else {
          scheduleConfigure()
        }
      },

      replaceConfig: (cfg) => {
        set({ config: cfg })
        scheduleConfigure(true)
      },

      play: () => {
        set({ selectedStep: null })
        send({ type: 'start' })
      },
      pause: () => send({ type: 'pause' }),
      toggle: () => (get().running ? get().pause() : get().play()),
      step: (n) => {
        set({ selectedStep: null })
        send({ type: 'step', n })
      },
      reset: () => {
        wantRunAfterBuild = false
        send({ type: 'reset' })
      },
      setSpeed: (s) => {
        set({ speed: s })
        send({ type: 'setSpeed', stepsPerTick: s })
      },

      scrubTo: (index) => {
        const snap = history.at(index)
        if (!snap) return
        if (get().running) send({ type: 'pause' })
        set({ selectedStep: snap.step })
      },
      jumpToStep: (step) => {
        const i = history.nearest(step, (s) => s.step)
        if (i >= 0) get().scrubTo(i)
      },
      goLive: () => set({ selectedStep: null }),
      nudge: (delta) => {
        const { selectedStep } = get()
        const cur =
          selectedStep === null ? history.length - 1 : history.nearest(selectedStep, (s) => s.step)
        const next = Math.max(0, Math.min(history.length - 1, cur + delta))
        if (delta > 0 && next === history.length - 1 && selectedStep === null) return
        get().scrubTo(next)
      },

      applyFix: (fix) => {
        const cfg = { ...get().config, ...fix.patch }
        const keys = Object.keys(fix.patch) as (keyof PlaygroundConfig)[]
        const liveOnly = !fix.reset && keys.every((k) => LIVE_KEYS.has(k))
        set({ config: cfg })
        if (liveOnly) {
          send({ type: 'live', lr: fix.patch.lr, l2: fix.patch.l2, batchSize: fix.patch.batchSize })
          runDoctor()
        } else {
          wantRunAfterBuild = true
          scheduleConfigure(true)
        }
        useUI.getState().flash('loss')
      },
    }
  })

  /** The snapshot every view should render: the live head, or the one the user scrubbed to. */
  const viewSnapshot = (selectedStep: number | null): Snapshot | undefined => {
    if (selectedStep === null) return history.last()
    return history.at(history.nearest(selectedStep, (x) => x.step))
  }

  return { useRun, history, viewSnapshot }
}

// The playground's shared run, connected to the shared worker.
const playground = createRunStore({ send: defaultSend, onMessage: defaultOnMessage })
export const useRun = playground.useRun
export const history = playground.history
export const viewSnapshot = playground.viewSnapshot
