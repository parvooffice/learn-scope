/// <reference lib="webworker" />
import type * as TFNS from '@tensorflow/tfjs'
import type { MnistTrainer, MnistDataset } from '../engine/tf/mnistTrainer'
import type { FromMnistWorker, MnistConfig, MnistSnapshot, ToMnistWorker } from './mnistProtocol'
import { initTf } from '../engine/tf/backend'

const post = (m: FromMnistWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer)

let tf: typeof TFNS | null = null
let Ctor: typeof MnistTrainer | null = null
let data: MnistDataset | null = null
let config: MnistConfig | null = null
let trainer: MnistTrainer | null = null
let focusPixels: Float32Array | null = null
let running = false
let stepsPerTick = 4
let tickCount = 0
let loop: Promise<void> = Promise.resolve()
let token = 0
const EVAL_EVERY = 6

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

async function init() {
  if (tf) return
  const i = await initTf()
  tf = i.tf
  Ctor = (await import('../engine/tf/mnistTrainer')).MnistTrainer
  post({ type: 'ready', backend: i.backend, notes: i.notes })
}

async function stopLoop() {
  running = false
  token++
  await loop
}

function emit(s: MnistSnapshot) {
  const buffers: Transferable[] = [s.filters.data.buffer]
  if (s.focus?.conv1) buffers.push(s.focus.conv1.data.buffer)
  if (s.focus?.conv2) buffers.push(s.focus.conv2.data.buffer)
  post({ type: 'snapshot', snapshot: s }, buffers)
}

async function build() {
  await stopLoop()
  await init()
  if (!data || !config) return
  trainer?.dispose()
  trainer = new Ctor!(tf!, config, data)
  trainer.setFocus(focusPixels)
  tickCount = 0
  post({ type: 'status', running: false })
  post({ type: 'built', model: config.model, params: trainer.params })
  emit(await trainer.snapshot(true))
}

function start() {
  if (!trainer || running) return
  running = true
  const my = ++token
  post({ type: 'status', running: true })
  loop = (async () => {
    while (running && my === token && trainer) {
      const t0 = performance.now()
      try {
        for (let i = 0; i < stepsPerTick; i++) await trainer.trainStep()
        tickCount++
        emit(await trainer.snapshot(tickCount % EVAL_EVERY === 0))
      } catch (e) {
        running = false
        post({ type: 'error', message: String((e as Error)?.message ?? e) })
        break
      }
      await sleep(Math.max(0, 16 - (performance.now() - t0)))
    }
    if (my === token) {
      running = false
      post({ type: 'status', running: false })
    }
  })()
}

let queue: Promise<void> = Promise.resolve()
self.onmessage = (ev: MessageEvent<ToMnistWorker>) => {
  const msg = ev.data
  queue = queue
    .then(() => handle(msg))
    .catch((e) => post({ type: 'error', message: String(e?.message ?? e) }))
}

async function handle(msg: ToMnistWorker) {
  switch (msg.type) {
    case 'data':
      data = { images: msg.images, labels: msg.labels, nTrain: msg.nTrain, nTest: msg.nTest }
      if (config) await build()
      break
    case 'configure':
      config = msg.config
      if (data) await build()
      break
    case 'start':
      start()
      break
    case 'pause':
      await stopLoop()
      post({ type: 'status', running: false })
      if (trainer) emit(await trainer.snapshot(true))
      break
    case 'step':
      if (!trainer) break
      await stopLoop()
      for (let i = 0; i < msg.n; i++) await trainer.trainStep()
      emit(await trainer.snapshot(true))
      post({ type: 'status', running: false })
      break
    case 'setSpeed':
      stepsPerTick = Math.max(1, Math.round(msg.stepsPerTick))
      break
    case 'focus': {
      focusPixels = msg.pixels
      if (trainer) {
        trainer.setFocus(msg.pixels)
        // answer immediately so drawing feels live, even while paused
        const focus = await trainer.focus(msg.pixels)
        const buffers: Transferable[] = []
        if (focus.conv1) buffers.push(focus.conv1.data.buffer)
        if (focus.conv2) buffers.push(focus.conv2.data.buffer)
        post({ type: 'focus', focus }, buffers)
      }
      break
    }
  }
}
