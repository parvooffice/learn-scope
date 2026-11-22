import { describe, expect, it } from 'vitest'
import { PRESETS } from './runs'
import { DEFAULT_CONFIG } from '../../engine/types'
import { sanitizeConfig } from '../../lib/urlConfig'

describe('Compare presets', () => {
  for (const p of PRESETS) {
    it(`${p.id}: both configs are valid (nothing gets clamped or repaired)`, () => {
      for (const side of [p.a, p.b]) {
        const cfg = { ...DEFAULT_CONFIG, ...p.shared, ...side }
        expect(sanitizeConfig(cfg)).toEqual(cfg)
      }
    })
    it(`${p.id}: A and B actually differ, and only in what the label says`, () => {
      expect(JSON.stringify(p.a)).not.toEqual(JSON.stringify(p.b))
      // data settings are shared, so neither side may override them
      for (const k of ['dataset', 'noise', 'samples', 'testSplit', 'seed'] as const) {
        expect(p.a).not.toHaveProperty(k)
        expect(p.b).not.toHaveProperty(k)
      }
    })
  }
  it('has unique ids', () => {
    expect(new Set(PRESETS.map((p) => p.id)).size).toBe(PRESETS.length)
  })
})
