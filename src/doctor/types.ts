import type { PlaygroundConfig, Snapshot } from '../engine/types'

export type Severity = 'info' | 'warn' | 'critical'
export type ChartId = 'loss' | 'grad' | 'weights' | 'activations' | 'network'

export interface Fix {
  label: string
  patch: Partial<PlaygroundConfig>
  /** true when the run must restart (e.g. architecture change or NaNs in the weights) */
  reset: boolean
}

export interface Diagnosis {
  /** stable id so the UI can keep the same card between runs */
  id: string
  severity: Severity
  title: string
  /** plain-English explanation: what is happening and why it matters */
  explanation: string
  /** where to look: jump to this training step and highlight this chart */
  evidence: { step: number; chart: ChartId; detail: string }
  fix?: Fix
}

export interface DoctorContext {
  history: readonly Snapshot[]
  config: PlaygroundConfig
}

/** A rule is a pure function: same history in, same diagnosis out. */
export type Rule = (ctx: DoctorContext) => Diagnosis | null
