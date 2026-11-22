import type { Snapshot } from '../engine/types'

export const last = <T>(a: readonly T[], k = 1): T[] => a.slice(Math.max(0, a.length - k))
export const avg = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0)

/** Round a learning rate to 2 significant digits so suggestions look human ("0.01", not "0.0100003"). */
export function niceLr(x: number): number {
  const clamped = Math.min(3, Math.max(0.0001, x))
  return Number(clamped.toPrecision(2))
}

export function isBad(n: number): boolean {
  return !Number.isFinite(n)
}

export function snapshotHasNaN(s: Snapshot): boolean {
  if (isBad(s.trainLoss) || isBad(s.testLoss)) return true
  for (const w of s.weights) {
    for (let i = 0; i < w.W.length; i++) if (isBad(w.W[i])) return true
  }
  return false
}

export const fmt = (x: number, digits = 3) =>
  Math.abs(x) < 1e-3 && x !== 0 ? x.toExponential(1) : Number(x.toPrecision(digits)).toString()
