import { AutoCanvas } from '../components/AutoCanvas'
import { cssVar, rgb, viridis } from '../lib/colors'

/** 10×10 confusion matrix: rows are the true digit, columns the predicted digit. */
export function ConfusionMatrix({ confusion }: { confusion: number[] }) {
  const draw = (ctx: CanvasRenderingContext2D, w: number) => {
    const pad = 22
    const cell = (w - pad) / 10
    const text = cssVar('--muted', '#8b98a8')
    // colour by row-normalised rate so every class is readable
    ctx.font = '10px ui-monospace, Consolas, monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    for (let t = 0; t < 10; t++) {
      let rowSum = 0
      for (let p = 0; p < 10; p++) rowSum += confusion[t * 10 + p]
      for (let p = 0; p < 10; p++) {
        const v = confusion[t * 10 + p]
        const frac = rowSum ? v / rowSum : 0
        ctx.fillStyle = rgb(viridis(Math.sqrt(frac)))
        ctx.fillRect(pad + p * cell, pad + t * cell, cell - 1, cell - 1)
        if (v > 0) {
          ctx.fillStyle = frac > 0.55 ? '#05070a' : '#ffffff'
          ctx.fillText(String(v), pad + p * cell + cell / 2, pad + t * cell + cell / 2)
        }
      }
      ctx.fillStyle = text
      ctx.fillText(String(t), pad / 2, pad + t * cell + cell / 2)
      ctx.fillText(String(t), pad + t * cell + cell / 2, pad / 2)
    }
  }
  return (
    <AutoCanvas
      draw={draw}
      height={Math.round(300)}
      label="Confusion matrix: rows are the true digit, columns the digit the model predicted. A bright diagonal means mostly correct."
    />
  )
}
