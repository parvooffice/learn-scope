import type { PlaygroundConfig, Snapshot } from '../engine/types'

/** Messages the UI sends to the training worker. */
export type ToWorker =
  | { type: 'configure'; config: PlaygroundConfig }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'step'; n: number }
  | { type: 'reset' }
  | { type: 'setSpeed'; stepsPerTick: number }
  | { type: 'live'; lr?: number; l2?: number; batchSize?: number }

/** Messages the worker sends back. */
export type FromWorker =
  | { type: 'ready'; backend: string; notes: string[] }
  /** a fresh model was built: everything before this message belongs to the previous run */
  | { type: 'built' }
  | { type: 'snapshot'; snapshot: Snapshot }
  | { type: 'status'; running: boolean }
  | { type: 'error'; message: string }
