import { AutoCanvas } from '../components/AutoCanvas'
import { viridis } from '../lib/colors'
import { paintField, GRID } from './DecisionBoundary'

/** What does one neuron respond to? Its activation over the whole 2-D input plane. */
export function NeuronPreview({
  grid,
  nout,
  unit,
  size = 120,
}: {
  /** that layer's activations on the grid: GRID*GRID cells, `nout` units per cell */
  grid: Float32Array
  nout: number
  unit: number
  size?: number
}) {
  const values = new Float32Array(GRID * GRID)
  let lo = Infinity
  let hi = -Infinity
  for (let c = 0; c < GRID * GRID; c++) {
    const v = grid[c * nout + unit]
    values[c] = v
    lo = Math.min(lo, v)
    hi = Math.max(hi, v)
  }
  const span = Math.max(hi - lo, 1e-6)
  for (let c = 0; c < values.length; c++) values[c] = (values[c] - lo) / span
  return (
    <div>
      <AutoCanvas
        fixedWidth={size}
        height={size}
        label={`Preview of one neuron: brighter means this neuron fires more strongly at that position. Range ${lo.toFixed(2)} to ${hi.toFixed(2)}.`}
        draw={(ctx, w, h) => paintField(ctx, values, GRID, 0, 0, w, h, viridis)}
      />
      <div className="num mt-1 flex justify-between text-[10px] text-muted">
        <span>min {lo.toFixed(2)}</span>
        <span>max {hi.toFixed(2)}</span>
      </div>
    </div>
  )
}
