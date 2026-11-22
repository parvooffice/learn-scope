import type { Dataset, DatasetKind } from '../types'
import { gaussian, mulberry32 } from '../../lib/rng'

/** All generators return points that sit roughly inside [-1.2, 1.2]^2. */
export function generateDataset(
  kind: DatasetKind,
  n: number,
  noise: number,
  seed: number,
): Dataset {
  const rng = mulberry32(seed * 7919 + 13)
  const x = new Float32Array(n * 2)
  const y = new Uint8Array(n)
  const jitter = () => gaussian(rng) * noise

  for (let i = 0; i < n; i++) {
    const cls = i % 2
    y[i] = cls
    let px = 0
    let py = 0
    switch (kind) {
      case 'moons': {
        const t = rng() * Math.PI
        if (cls === 0) {
          px = Math.cos(t)
          py = Math.sin(t)
        } else {
          px = 1 - Math.cos(t)
          py = 0.5 - Math.sin(t)
        }
        px = (px - 0.5) * 0.85 + jitter() * 0.5
        py = (py - 0.25) * 1.3 + jitter() * 0.5
        break
      }
      case 'circles': {
        const r = cls === 0 ? 0.42 : 0.88
        const t = rng() * Math.PI * 2
        px = r * Math.cos(t) + jitter() * 0.5
        py = r * Math.sin(t) + jitter() * 0.5
        break
      }
      case 'spiral': {
        const t = 0.08 + rng() * 0.92
        const theta = t * 1.75 * Math.PI * 2 + cls * Math.PI
        px = t * 0.95 * Math.cos(theta) + jitter() * 0.5
        py = t * 0.95 * Math.sin(theta) + jitter() * 0.5
        break
      }
      case 'xor': {
        px = rng() * 2 - 1
        py = rng() * 2 - 1
        // push points away from the axes so the problem is learnable
        px += Math.sign(px) * 0.08
        py += Math.sign(py) * 0.08
        y[i] = px > 0 !== py > 0 ? 1 : 0
        px += jitter() * 0.6
        py += jitter() * 0.6
        break
      }
      case 'blobs': {
        const c = cls === 0 ? -0.5 : 0.5
        px = c + gaussian(rng) * (0.12 + noise * 0.8)
        py = c + gaussian(rng) * (0.12 + noise * 0.8)
        break
      }
    }
    x[i * 2] = px
    x[i * 2 + 1] = py
  }
  return { x, y, n }
}

/** Deterministic shuffle-then-split into train and test sets. */
export function splitDataset(
  ds: Dataset,
  testFrac: number,
  seed: number,
): { train: Dataset; test: Dataset } {
  const rng = mulberry32(seed * 104729 + 5)
  const idx = Array.from({ length: ds.n }, (_, i) => i)
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1))
    ;[idx[i], idx[j]] = [idx[j], idx[i]]
  }
  const nTest = Math.max(1, Math.round(ds.n * testFrac))
  const pick = (ids: number[]): Dataset => {
    const x = new Float32Array(ids.length * 2)
    const y = new Uint8Array(ids.length)
    ids.forEach((id, k) => {
      x[k * 2] = ds.x[id * 2]
      x[k * 2 + 1] = ds.x[id * 2 + 1]
      y[k] = ds.y[id]
    })
    return { x, y, n: ids.length }
  }
  return { test: pick(idx.slice(0, nTest)), train: pick(idx.slice(nTest)) }
}

export const DATASET_LABELS: Record<DatasetKind, string> = {
  moons: 'Two moons',
  circles: 'Circles',
  spiral: 'Spiral',
  xor: 'XOR',
  blobs: 'Gaussian blobs',
}
