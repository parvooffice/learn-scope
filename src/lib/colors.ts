/** Perceptually uniform viridis, sampled at 9 stops, plus a blue/red diverging scale for signed values. */
const VIRIDIS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
]

export function viridis(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1)
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x))
  const f = x - i
  const a = VIRIDIS[i]
  const b = VIRIDIS[i + 1]
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/** Diverging scale: t in [-1, 1]; negative = blue, positive = red, 0 = neutral grey. */
export function diverging(
  t: number,
  neutral: [number, number, number] = [235, 238, 242],
): [number, number, number] {
  const x = Math.min(1, Math.max(-1, t))
  const neg: [number, number, number] = [60, 120, 255]
  const pos: [number, number, number] = [240, 80, 70]
  const target = x < 0 ? neg : pos
  const f = Math.abs(x)
  return [
    neutral[0] + (target[0] - neutral[0]) * f,
    neutral[1] + (target[1] - neutral[1]) * f,
    neutral[2] + (target[2] - neutral[2]) * f,
  ]
}

export const rgb = (c: [number, number, number], a = 1) =>
  `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`

/** Read a CSS custom property (theme colour) at draw time. */
export function cssVar(name: string, fallback = '#888'): string {
  if (typeof document === 'undefined') return fallback
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim()
  return v || fallback
}

/** Distinct, colour-blind-friendly series colours for per-layer lines. */
export const SERIES = ['#2dd4bf', '#f5b942', '#a78bfa', '#f0625c', '#4f8cff', '#34d399', '#f472b6']
