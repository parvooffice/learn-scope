import { create } from 'zustand'
import { loadMnist, type MnistData, PIXELS } from '../engine/datasets/mnist'
import { onMnistMessage, sendMnist } from '../engine/mnistClient'
import {
  DEFAULT_MNIST,
  type MnistConfig,
  type MnistEval,
  type MnistFocus,
  type MnistSnapshot,
} from '../worker/mnistProtocol'

export interface MnistPoint {
  step: number
  epoch: number
  trainLoss: number
  trainAcc: number
  testLoss?: number
  testAcc?: number
}

/** curve history lives outside React state (it grows every tick) */
export const mnistHistory: MnistPoint[] = []

export type FocusSource = { kind: 'test'; index: number } | { kind: 'draw' }

interface MnistState {
  config: MnistConfig
  loadStatus: string
  loaded: boolean
  fromCache: boolean
  ready: boolean
  backend: string | null
  notes: string[]
  params: number
  running: boolean
  error: string | null
  snapshot: MnistSnapshot | null
  lastEval: MnistEval | null
  focus: MnistFocus | null
  focusSource: FocusSource
  drawPixels: Float32Array | null
  data: MnistData | null
  speed: number
  version: number
  /** seconds to complete the first full epoch, measured on this machine */
  epochSeconds: number | null

  boot: () => void
  setConfig: (p: Partial<MnistConfig>) => void
  play: () => void
  pause: () => void
  step: (n: number) => void
  reset: () => void
  setSpeed: (n: number) => void
  focusTest: (index: number) => void
  focusDrawn: (pixels: Float32Array) => void
}

let booted = false
let wantRun = false

export function testPixels(data: MnistData, index: number): Float32Array {
  const out = new Float32Array(PIXELS)
  const off = (data.nTrain + index) * PIXELS
  for (let i = 0; i < PIXELS; i++) out[i] = data.images[off + i] / 255
  return out
}

export const useMnist = create<MnistState>((set, get) => ({
  config: DEFAULT_MNIST,
  loadStatus: 'Starting…',
  loaded: false,
  fromCache: false,
  ready: false,
  backend: null,
  notes: [],
  params: 0,
  running: false,
  error: null,
  snapshot: null,
  lastEval: null,
  focus: null,
  focusSource: { kind: 'test', index: 0 },
  drawPixels: null,
  data: null,
  speed: 4,
  version: 0,
  epochSeconds: null,

  boot: () => {
    if (booted) return
    booted = true
    onMnistMessage((m) => {
      switch (m.type) {
        case 'ready':
          set({ ready: true, backend: m.backend, notes: m.notes })
          break
        case 'built':
          mnistHistory.length = 0
          set((s) => ({
            params: m.params,
            snapshot: null,
            lastEval: null,
            epochSeconds: null,
            version: s.version + 1,
          }))
          if (wantRun) {
            wantRun = false
            sendMnist({ type: 'start' })
          }
          break
        case 'snapshot': {
          const s = m.snapshot
          mnistHistory.push({
            step: s.step,
            epoch: s.epoch,
            trainLoss: s.trainLoss,
            trainAcc: s.trainAcc,
            testLoss: s.eval?.testLoss,
            testAcc: s.eval?.testAcc,
          })
          set((st) => ({
            snapshot: s,
            lastEval: s.eval ?? st.lastEval,
            focus: s.focus ?? st.focus,
            version: st.version + 1,
            epochSeconds: st.epochSeconds ?? (s.epoch >= 1 ? s.elapsed / s.epoch : null),
          }))
          break
        }
        case 'focus':
          set({ focus: m.focus })
          break
        case 'status':
          set({ running: m.running })
          break
        case 'error':
          set({ error: m.message, running: false })
          break
      }
    })
    loadMnist((s) => set({ loadStatus: s }))
      .then(({ data, fromCache }) => {
        set({
          data,
          loaded: true,
          fromCache,
          loadStatus: fromCache ? 'Loaded from browser cache' : 'Downloaded and cached',
        })
        // the worker gets its own copy; keep ours for drawing test images
        sendMnist({
          type: 'data',
          images: data.images.slice(),
          labels: data.labels.slice(),
          nTrain: data.nTrain,
          nTest: data.nTest,
        })
        sendMnist({ type: 'setSpeed', stepsPerTick: get().speed })
        sendMnist({ type: 'configure', config: get().config })
        get().focusTest(0)
      })
      .catch((e) => set({ error: String(e?.message ?? e), loadStatus: 'Failed to load data' }))
  },

  setConfig: (p) => {
    const config = { ...get().config, ...p }
    wantRun = get().running
    set({ config })
    sendMnist({ type: 'configure', config })
  },
  play: () => sendMnist({ type: 'start' }),
  pause: () => sendMnist({ type: 'pause' }),
  step: (n) => sendMnist({ type: 'step', n }),
  reset: () => {
    wantRun = false
    sendMnist({ type: 'configure', config: { ...get().config, seed: get().config.seed + 1 } })
    set({ config: { ...get().config, seed: get().config.seed + 1 } })
  },
  setSpeed: (n) => {
    set({ speed: n })
    sendMnist({ type: 'setSpeed', stepsPerTick: n })
  },
  focusTest: (index) => {
    const { data } = get()
    if (!data) return
    set({ focusSource: { kind: 'test', index } })
    sendMnist({ type: 'focus', pixels: testPixels(data, index) })
  },
  focusDrawn: (pixels) => {
    set({ focusSource: { kind: 'draw' }, drawPixels: pixels })
    sendMnist({ type: 'focus', pixels: pixels.slice() })
  },
}))
