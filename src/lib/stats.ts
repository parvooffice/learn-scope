import type { Hist } from '../engine/types'

export const HIST_BINS = 24

export function meanStd(v: ArrayLike<number>): { mean: number; std: number } {
  const n = v.length
  if (n === 0) return { mean: 0, std: 0 }
  let s = 0
  for (let i = 0; i < n; i++) s += v[i]
  const mean = s / n
  let q = 0
  for (let i = 0; i < n; i++) q += (v[i] - mean) * (v[i] - mean)
  return { mean, std: Math.sqrt(q / n) }
}

export function meanAbs(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += Math.abs(v[i])
  return v.length ? s / v.length : 0
}

export function l2Norm(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s)
}

export function absMax(v: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < v.length; i++) m = Math.max(m, Math.abs(v[i]))
  return m
}

/** Fixed-bin histogram. If lo/hi are omitted they come from the data (with a tiny pad). */
export function histogram(v: ArrayLike<number>, bins = HIST_BINS, lo?: number, hi?: number): Hist {
  let min = lo ?? Infinity
  let max = hi ?? -Infinity
  if (lo === undefined || hi === undefined) {
    for (let i = 0; i < v.length; i++) {
      if (lo === undefined) min = Math.min(min, v[i])
      if (hi === undefined) max = Math.max(max, v[i])
    }
  }
  if (!isFinite(min) || !isFinite(max)) {
    min = 0
    max = 1
  }
  if (max - min < 1e-12) {
    min -= 0.5
    max += 0.5
  }
  const counts = new Array<number>(bins).fill(0)
  const w = (max - min) / bins
  for (let i = 0; i < v.length; i++) {
    let b = Math.floor((v[i] - min) / w)
    if (b < 0) b = 0
    if (b >= bins) b = bins - 1
    counts[b]++
  }
  return { counts, lo: min, hi: max }
}
