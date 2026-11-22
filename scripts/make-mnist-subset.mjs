// Builds the MNIST subset shipped in public/data from the public "model-builder" sprite
// (https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png + mnist_labels_uint8).
//
// usage: node scripts/make-mnist-subset.mjs <mnist_images.png> <mnist_labels_uint8> <outDir>
//
// The sprite is 784 pixels wide and 65000 rows tall: one 28x28 image per row. The first 55000
// rows are the training set, the last 10000 the test set (both already shuffled).
// We keep 10,000 train + 2,000 test images and store them as one 8-bit grayscale PNG
// (lossless, compresses well) plus a tiny label file.
import fs from 'node:fs'
import path from 'node:path'
import { PNG } from 'pngjs'

const [, , imgPath, labelPath, outDir] = process.argv
if (!imgPath || !labelPath || !outDir) {
  console.error('usage: node scripts/make-mnist-subset.mjs <images.png> <labels> <outDir>')
  process.exit(1)
}
const N_TRAIN = 10000
const N_TEST = 2000
const TRAIN_START = 0
const TEST_START = 55000

const src = PNG.sync.read(fs.readFileSync(imgPath))
if (src.width !== 784 || src.height < 65000)
  throw new Error(`unexpected sprite size ${src.width}x${src.height}`)
const labels = fs.readFileSync(labelPath)
if (labels.length !== 65000 * 10) throw new Error('unexpected label file size')

const rows = [
  ...Array.from({ length: N_TRAIN }, (_, i) => TRAIN_START + i),
  ...Array.from({ length: N_TEST }, (_, i) => TEST_START + i),
]
const out = new PNG({
  width: 784,
  height: rows.length,
  colorType: 0,
  inputColorType: 0,
  inputHasAlpha: false,
})
const lab = Buffer.alloc(rows.length)
rows.forEach((row, k) => {
  for (let x = 0; x < 784; x++) out.data[k * 784 + x] = src.data[(row * 784 + x) * 4] // red channel = gray
  let cls = -1
  for (let c = 0; c < 10; c++) if (labels[row * 10 + c] === 1) cls = c
  if (cls < 0) throw new Error(`row ${row} has no label`)
  lab[k] = cls
})
// pngjs with colorType 0 expects 1 byte per pixel when inputColorType is 0
fs.mkdirSync(outDir, { recursive: true })
const buf = PNG.sync.write(out, { colorType: 0, inputColorType: 0, inputHasAlpha: false })
fs.writeFileSync(path.join(outDir, 'mnist-12k.png'), buf)
fs.writeFileSync(path.join(outDir, 'mnist-12k-labels.bin'), lab)
console.log(
  `wrote ${rows.length} images (${(buf.length / 1024).toFixed(0)} KB png) and ${lab.length} labels`,
)
const counts = Array(10).fill(0)
lab.forEach((c) => counts[c]++)
console.log('class counts', counts.join(' '))
