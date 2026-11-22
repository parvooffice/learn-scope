import type { FromMnistWorker, ToMnistWorker } from '../worker/mnistProtocol'

let worker: Worker | null = null
const listeners = new Set<(m: FromMnistWorker) => void>()

function ensure(): Worker {
  if (!worker) {
    worker = new Worker(new URL('../worker/mnist.worker.ts', import.meta.url), { type: 'module' })
    worker.onmessage = (e: MessageEvent<FromMnistWorker>) => listeners.forEach((l) => l(e.data))
    worker.onerror = (e) =>
      listeners.forEach((l) => l({ type: 'error', message: e.message || 'MNIST worker crashed' }))
  }
  return worker
}

export function sendMnist(msg: ToMnistWorker, transfer: Transferable[] = []) {
  ensure().postMessage(msg, transfer)
}

export function onMnistMessage(fn: (m: FromMnistWorker) => void) {
  listeners.add(fn)
  ensure()
  return () => listeners.delete(fn)
}
