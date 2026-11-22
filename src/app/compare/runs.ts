import { createWorkerClient } from '../../engine/client'
import { createRunStore } from '../../store/run'
import type { PlaygroundConfig } from '../../engine/types'

export type RunHandle = ReturnType<typeof createRunStore> & { terminate: () => void }

/** Create one independent experiment: its own worker, history and diagnoses. */
export function createRun(initial: Partial<PlaygroundConfig>): RunHandle {
  const client = createWorkerClient()
  const run = createRunStore(client, initial)
  run.useRun.getState().boot()
  return { ...run, terminate: client.terminate }
}

export interface Preset {
  id: string
  label: string
  blurb: string
  shared: Partial<PlaygroundConfig>
  a: Partial<PlaygroundConfig>
  b: Partial<PlaygroundConfig>
}

const net = (units: number[], activation: 'relu' | 'tanh' | 'sigmoid' | 'gelu') =>
  units.map((u) => ({ units: u, activation }))

/** Ready-made comparisons that each teach one thing. */
export const PRESETS: Preset[] = [
  {
    id: 'optimizer',
    label: 'Adam vs SGD',
    blurb:
      'Same network, same data. Adam adapts a step size per weight; plain SGD uses one fixed step.',
    shared: { dataset: 'spiral', noise: 0.08, samples: 400 },
    a: { layers: net([8, 8], 'tanh'), optimizer: 'adam', lr: 0.03 },
    b: { layers: net([8, 8], 'tanh'), optimizer: 'sgd', lr: 0.1 },
  },
  {
    id: 'activation',
    label: 'ReLU vs tanh',
    blurb:
      'Different activation functions give different boundary shapes and different gradient flow.',
    shared: { dataset: 'moons', noise: 0.15, samples: 400 },
    a: { layers: net([8, 8, 8], 'tanh'), optimizer: 'adam', lr: 0.02 },
    b: { layers: net([8, 8, 8], 'relu'), optimizer: 'adam', lr: 0.02 },
  },
  {
    id: 'lr',
    label: 'Sensible vs huge learning rate',
    blurb: 'Watch run B blow up and the Model Doctor diagnose it, while run A trains normally.',
    shared: { dataset: 'circles', noise: 0.08, samples: 400 },
    a: { layers: net([8, 8], 'tanh'), optimizer: 'adam', lr: 0.01 },
    b: { layers: net([8, 8], 'tanh'), optimizer: 'adam', lr: 1.5 },
  },
  {
    id: 'size',
    label: 'Small vs big network',
    blurb: 'Two hidden neurons cannot bend enough for the spiral; a larger net can.',
    shared: { dataset: 'spiral', noise: 0.05, samples: 400 },
    a: { layers: net([3], 'tanh'), optimizer: 'adam', lr: 0.03 },
    b: { layers: net([12, 12], 'tanh'), optimizer: 'adam', lr: 0.03 },
  },
  {
    id: 'l2',
    label: 'No regularization vs L2',
    blurb:
      'On noisy data a big network memorises. L2 weight decay (run B) should narrow the train/test gap.',
    shared: { dataset: 'moons', noise: 0.35, samples: 200, testSplit: 0.4 },
    a: { layers: net([16, 16, 16], 'tanh'), optimizer: 'adam', lr: 0.02, l2: 0 },
    b: { layers: net([16, 16, 16], 'tanh'), optimizer: 'adam', lr: 0.02, l2: 0.02 },
  },
]
