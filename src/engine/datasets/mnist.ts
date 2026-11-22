import { openDB } from 'idb'

export const MNIST_N_TRAIN = 10000
export const MNIST_N_TEST = 2000
export const PIXELS = 784

export interface MnistData {
  /** n × 784 grayscale bytes, 0..255; first N_TRAIN images are train, the rest test */
  images: Uint8Array
  labels: Uint8Array
  nTrain: number
  nTest: number
}

const DB = 'learnscope'
const STORE = 'datasets'
const KEY = 'mnist-12k-v1'

async function readCache(): Promise<MnistData | null> {
  try {
    const db = await openDB(DB, 1, { upgrade: (d) => void d.createObjectStore(STORE) })
    const hit = (await db.get(STORE, KEY)) as MnistData | undefined
    if (hit && hit.images.length === (hit.nTrain + hit.nTest) * PIXELS) return hit
  } catch {
    /* IndexedDB unavailable (private mode etc.): just download */
  }
  return null
}

async function writeCache(d: MnistData) {
  try {
    const db = await openDB(DB, 1, { upgrade: (x) => void x.createObjectStore(STORE) })
    await db.put(STORE, d, KEY)
  } catch {
    /* ignore */
  }
}

async function decodePng(blob: Blob, total: number): Promise<Uint8Array> {
  const bmp = await createImageBitmap(blob)
  if (bmp.width !== PIXELS || bmp.height !== total)
    throw new Error(`unexpected MNIST image size ${bmp.width}×${bmp.height}`)
  const canvas = document.createElement('canvas')
  canvas.width = bmp.width
  canvas.height = bmp.height
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bmp, 0, 0)
  const rgba = ctx.getImageData(0, 0, bmp.width, bmp.height).data
  const out = new Uint8Array(total * PIXELS)
  for (let i = 0; i < out.length; i++) out[i] = rgba[i * 4]
  return out
}

/**
 * Load the MNIST subset. First visit downloads a 2.3 MB PNG from our own site (decoded in the
 * browser); afterwards the decoded pixels come straight from IndexedDB.
 */
export async function loadMnist(
  onStatus?: (s: string) => void,
): Promise<{ data: MnistData; fromCache: boolean }> {
  onStatus?.('Checking cache…')
  const cached = await readCache()
  if (cached) return { data: cached, fromCache: true }
  const total = MNIST_N_TRAIN + MNIST_N_TEST
  const base = import.meta.env.BASE_URL
  onStatus?.('Downloading MNIST subset (2.3 MB)…')
  const [imgRes, labRes] = await Promise.all([
    fetch(`${base}data/mnist-12k.png`),
    fetch(`${base}data/mnist-12k-labels.bin`),
  ])
  if (!imgRes.ok || !labRes.ok) throw new Error('Could not download the MNIST data files.')
  onStatus?.('Decoding images…')
  const images = await decodePng(await imgRes.blob(), total)
  const labels = new Uint8Array(await labRes.arrayBuffer())
  if (labels.length !== total) throw new Error('MNIST label file has the wrong size.')
  const data: MnistData = { images, labels, nTrain: MNIST_N_TRAIN, nTest: MNIST_N_TEST }
  void writeCache(data)
  return { data, fromCache: false }
}

/** 28×28 → centred, MNIST-style (centre of mass in the middle) pixels from a drawing canvas. */
export function canvasToMnist(src: HTMLCanvasElement): Float32Array {
  const small = document.createElement('canvas')
  small.width = 28
  small.height = 28
  const sctx = small.getContext('2d', { willReadFrequently: true })!
  sctx.fillStyle = '#000'
  sctx.fillRect(0, 0, 28, 28)
  // MNIST digits fit a 20×20 box, centred by centre of mass inside the 28×28 frame
  sctx.imageSmoothingEnabled = true
  sctx.imageSmoothingQuality = 'high'
  sctx.drawImage(src, 0, 0, src.width, src.height, 4, 4, 20, 20)
  const d = sctx.getImageData(0, 0, 28, 28).data
  const px = new Float32Array(784)
  let sum = 0
  let cx = 0
  let cy = 0
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      const v = d[(y * 28 + x) * 4] / 255
      px[y * 28 + x] = v
      sum += v
      cx += v * x
      cy += v * y
    }
  if (sum < 1e-3) return px
  const dx = Math.round(13.5 - cx / sum)
  const dy = Math.round(13.5 - cy / sum)
  const out = new Float32Array(784)
  for (let y = 0; y < 28; y++)
    for (let x = 0; x < 28; x++) {
      const sx = x - dx
      const sy = y - dy
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = px[sy * 28 + sx]
    }
  return out
}
