import type { FromWorker, ToWorker } from '../worker/protocol'

let worker: Worker | null = null
const listeners = new Set<(m: FromWorker) => void>()

function ensure(): Worker {
  if (!worker) {
    // The worker (and the whole TensorFlow.js bundle) is only fetched when the playground opens.
    worker = new Worker(new URL('../worker/training.worker.ts', import.meta.url), {
      type: 'module',
    })
    worker.onmessage = (e: MessageEvent<FromWorker>) => listeners.forEach((l) => l(e.data))
    worker.onerror = (e) =>
      listeners.forEach((l) => l({ type: 'error', message: e.message || 'Worker crashed' }))
  }
  return worker
}

export function send(msg: ToWorker) {
  ensure().postMessage(msg)
}

export function onMessage(fn: (m: FromWorker) => void) {
  listeners.add(fn)
  ensure()
  return () => listeners.delete(fn)
}

/** An independent worker connection (used by Compare, which runs two experiments at once). */
export function createWorkerClient() {
  const w = new Worker(new URL('../worker/training.worker.ts', import.meta.url), {
    type: 'module',
  })
  const subs = new Set<(m: FromWorker) => void>()
  w.onmessage = (e: MessageEvent<FromWorker>) => subs.forEach((l) => l(e.data))
  w.onerror = (e) =>
    subs.forEach((l) => l({ type: 'error', message: e.message || 'Worker crashed' }))
  return {
    send: (msg: ToWorker) => w.postMessage(msg),
    onMessage: (fn: (m: FromWorker) => void) => {
      subs.add(fn)
      return () => subs.delete(fn)
    },
    terminate: () => {
      subs.clear()
      w.terminate()
    },
  }
}
