/// <reference lib="webworker" />
import type * as TFNS from '@tensorflow/tfjs'
import type { Trainer } from '../engine/tf/trainer'
import type { FromWorker, ToWorker } from './protocol'
import type { PlaygroundConfig, Snapshot } from '../engine/types'
import { generateDataset, splitDataset } from '../engine/datasets/toy'

/**
 * The training worker owns the model. The UI thread never touches TensorFlow.js, so heavy
 * training can't make the interface janky.
 */
const post = (msg: FromWorker, transfer: Transferable[] = []) => self.postMessage(msg, transfer)

let tf: typeof TFNS | null = null
let TrainerCtor: typeof Trainer | null = null
let trainer: Trainer | null = null
let config: PlaygroundConfig | null = null
let running = false
let stepsPerTick = 4
let loop: Promise<void> = Promise.resolve()
let loopToken = 0
const MIN_TICK_MS = 33

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms))

async function initTf() {
  if (tf) return
  const notes: string[] = []
  const lib = await import('@tensorflow/tfjs')
  let webgpuLoaded = false
  try {
    await import('@tensorflow/tfjs-backend-webgpu')
    webgpuLoaded = true
  } catch (e) {
    notes.push(`WebGPU backend failed to load: ${String(e)}`)
  }
  const hasGpu = typeof navigator !== 'undefined' && 'gpu' in navigator
  for (const name of ['webgpu', 'webgl', 'cpu']) {
    if (name === 'webgpu' && !(webgpuLoaded && hasGpu)) {
      notes.push('WebGPU is not available in this browser.')
      continue
    }
    try {
      const ok = await lib.setBackend(name)
      await lib.ready()
      if (ok && lib.getBackend() === name) {
        tf = lib
        TrainerCtor = (await import('../engine/tf/trainer')).Trainer
        post({ type: 'ready', backend: name, notes })
        return
      }
      notes.push(`Backend ${name} could not be initialised.`)
    } catch (e) {
      notes.push(`Backend ${name} failed: ${String(e)}`)
    }
  }
  throw new Error('No TensorFlow.js backend could be initialised.')
}

async function stopLoop() {
  running = false
  loopToken++
  await loop
}

function emit(s: Snapshot) {
  const buffers: Transferable[] = []
  for (const w of s.weights) buffers.push(w.W.buffer, w.b.buffer)
  post({ type: 'snapshot', snapshot: s }, buffers)
}

async function build(cfg: PlaygroundConfig) {
  await stopLoop()
  await initTf()
  trainer?.dispose()
  config = cfg
  const ds = generateDataset(cfg.dataset, cfg.samples, cfg.noise, cfg.seed)
  const { train, test } = splitDataset(ds, cfg.testSplit, cfg.seed)
  trainer = new TrainerCtor!(tf!, cfg, train, test)
  post({ type: 'status', running: false })
  post({ type: 'built' })
  emit(await trainer.snapshot())
}

function startLoop() {
  if (!trainer || running) return
  running = true
  const my = ++loopToken
  post({ type: 'status', running: true })
  loop = (async () => {
    while (running && my === loopToken && trainer) {
      const t0 = performance.now()
      try {
        for (let i = 0; i < stepsPerTick; i++) trainer.trainStep()
        emit(await trainer.snapshot())
      } catch (e) {
        running = false
        post({ type: 'error', message: String(e) })
        break
      }
      const spent = performance.now() - t0
      await sleep(Math.max(0, MIN_TICK_MS - spent))
    }
    if (my === loopToken) {
      running = false
      post({ type: 'status', running: false })
    }
  })()
}

let queue: Promise<void> = Promise.resolve()
self.onmessage = (ev: MessageEvent<ToWorker>) => {
  const msg = ev.data
  // process messages one at a time so configure/step/start can't interleave
  queue = queue
    .then(() => handle(msg))
    .catch((e) => post({ type: 'error', message: String(e?.message ?? e) }))
}

async function handle(msg: ToWorker) {
  switch (msg.type) {
    case 'configure':
      await build(msg.config)
      break
    case 'reset':
      if (config) await build(config)
      break
    case 'start':
      startLoop()
      break
    case 'pause':
      await stopLoop()
      post({ type: 'status', running: false })
      break
    case 'step':
      if (!trainer) break
      await stopLoop()
      for (let i = 0; i < msg.n; i++) trainer.trainStep()
      emit(await trainer.snapshot())
      post({ type: 'status', running: false })
      break
    case 'setSpeed':
      stepsPerTick = Math.max(1, Math.round(msg.stepsPerTick))
      break
    case 'live':
      if (!trainer) break
      if (msg.lr !== undefined) trainer.lr = msg.lr
      if (msg.l2 !== undefined) trainer.l2 = msg.l2
      if (msg.batchSize !== undefined) trainer.batchSize = msg.batchSize
      break
  }
}
