import { AutoCanvas } from '../components/AutoCanvas'

let tmp: HTMLCanvasElement | null = null

/** Paint a 28×28 grayscale image (values 0..1) at a given size. */
export function DigitCanvas({
  pixels,
  size = 140,
  label,
}: {
  pixels: Float32Array | null
  size?: number
  label: string
}) {
  return (
    <AutoCanvas
      fixedWidth={size}
      height={size}
      label={label}
      style={{ borderRadius: 8, border: '1px solid var(--border)' }}
      draw={(ctx, w, h) => {
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, w, h)
        if (!pixels) return
        if (!tmp) tmp = document.createElement('canvas')
        tmp.width = 28
        tmp.height = 28
        const t = tmp.getContext('2d')!
        const img = t.createImageData(28, 28)
        for (let i = 0; i < 784; i++) {
          const v = Math.round(pixels[i] * 255)
          img.data[i * 4] = v
          img.data[i * 4 + 1] = v
          img.data[i * 4 + 2] = v
          img.data[i * 4 + 3] = 255
        }
        t.putImageData(img, 0, 0)
        ctx.imageSmoothingEnabled = false
        ctx.drawImage(tmp, 0, 0, w, h)
      }}
    />
  )
}
