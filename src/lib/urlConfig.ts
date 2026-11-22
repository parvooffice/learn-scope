import {
  DEFAULT_CONFIG,
  type Activation,
  type DatasetKind,
  type OptimizerKind,
  type PlaygroundConfig,
} from '../engine/types'

const DATASETS: DatasetKind[] = ['moons', 'circles', 'spiral', 'xor', 'blobs']
const ACTS: Activation[] = ['relu', 'tanh', 'sigmoid', 'gelu']
const OPTS: OptimizerKind[] = ['sgd', 'momentum', 'adam']

const clamp = (v: unknown, lo: number, hi: number, fallback: number) => {
  const n = typeof v === 'number' && isFinite(v) ? v : fallback
  return Math.min(hi, Math.max(lo, n))
}
const pick = <T extends string>(v: unknown, allowed: T[], fallback: T): T =>
  allowed.includes(v as T) ? (v as T) : fallback

/** Make an untrusted object into a valid config (unknown / out-of-range values are repaired). */
export function sanitizeConfig(raw: unknown): PlaygroundConfig {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const d = DEFAULT_CONFIG
  const layersRaw = Array.isArray(r.layers) ? r.layers : d.layers
  const layers = layersRaw.slice(0, 6).map((l) => {
    const o = (l ?? {}) as Record<string, unknown>
    return {
      units: Math.round(clamp(o.units, 1, 16, 4)),
      activation: pick(o.activation, ACTS, 'tanh'),
    }
  })
  return {
    dataset: pick(r.dataset, DATASETS, d.dataset),
    noise: clamp(r.noise, 0, 0.5, d.noise),
    samples: Math.round(clamp(r.samples, 100, 1000, d.samples)),
    testSplit: clamp(r.testSplit, 0.1, 0.5, d.testSplit),
    layers,
    lr: clamp(r.lr, 0.0001, 3, d.lr),
    batchSize: Math.round(clamp(r.batchSize, 1, 256, d.batchSize)),
    optimizer: pick(r.optimizer, OPTS, d.optimizer),
    l2: clamp(r.l2, 0, 0.1, d.l2),
    seed: Math.round(clamp(r.seed, 0, 999999, d.seed)),
  }
}

// btoa/atob exist in browsers, workers and Node >= 16. Our JSON is plain ASCII so no UTF-8 dance needed.
function toBase64Url(s: string): string {
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}
function fromBase64Url(s: string): string {
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/')
  return atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4))
}

/** Config -> short URL-safe string. */
export function encodeConfig(cfg: PlaygroundConfig): string {
  return toBase64Url(JSON.stringify(cfg))
}

/** URL-safe string -> config (or null if it can't be parsed). */
export function decodeConfig(s: string): PlaygroundConfig | null {
  try {
    return sanitizeConfig(JSON.parse(fromBase64Url(s)))
  } catch {
    return null
  }
}
