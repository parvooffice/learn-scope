import type * as TFNS from '@tensorflow/tfjs'

export interface TfInit {
  tf: typeof TFNS
  backend: string
  notes: string[]
}

let cached: Promise<TfInit> | null = null

/**
 * Load TensorFlow.js and pick the fastest available backend: WebGPU, then WebGL, then CPU.
 * Works inside a Web Worker. The result is cached so every worker initialises only once.
 */
export function initTf(): Promise<TfInit> {
  if (!cached) cached = doInit()
  return cached
}

async function doInit(): Promise<TfInit> {
  const notes: string[] = []
  const tf = await import('@tensorflow/tfjs')
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
      const ok = await tf.setBackend(name)
      await tf.ready()
      if (ok && tf.getBackend() === name) return { tf, backend: name, notes }
      notes.push(`Backend ${name} could not be initialised.`)
    } catch (e) {
      notes.push(`Backend ${name} failed: ${String(e)}`)
    }
  }
  throw new Error('No TensorFlow.js backend could be initialised.')
}
