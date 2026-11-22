import { useRun, viewSnapshot } from './run'
import { layerSpecs, type LayerSpec, type Snapshot } from '../engine/types'

/** True when a snapshot was produced by a network with this architecture. */
export function snapshotMatches(specs: LayerSpec[], s: Snapshot): boolean {
  if (s.weights.length !== specs.length || s.layers.length !== specs.length) return false
  return specs.every((sp, i) => s.weights[i].W.length === sp.nin * sp.nout)
}

/**
 * The snapshot every view should draw (live head, or the time-travel target).
 * Returns undefined while the worker is still rebuilding after an architecture change.
 */
export function useViewSnapshot(): Snapshot | undefined {
  const version = useRun((s) => s.version)
  const selected = useRun((s) => s.selectedStep)
  const config = useRun((s) => s.config)
  void version
  const snap = viewSnapshot(selected)
  return snap && snapshotMatches(layerSpecs(config), snap) ? snap : undefined
}
