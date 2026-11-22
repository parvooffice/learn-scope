/**
 * A bounded history for training snapshots.
 *
 * Two limits apply: a maximum number of items AND a maximum number of bytes.
 * When either is exceeded we do NOT just drop the oldest item (that would make "time travel"
 * forget the beginning of training). Instead we thin out the oldest half by keeping every
 * second item, so older history gets coarser while recent history stays dense.
 * The first and last items are always kept.
 */
export class HistoryBuffer<T> {
  private items: T[] = []
  private sizes: number[] = []
  private bytes = 0

  constructor(
    private readonly sizeOf: (t: T) => number,
    private readonly maxItems = 4000,
    private readonly maxBytes = 150 * 1024 * 1024,
  ) {}

  get length() {
    return this.items.length
  }
  get totalBytes() {
    return this.bytes
  }
  at(i: number): T | undefined {
    return this.items[i]
  }
  last(): T | undefined {
    return this.items[this.items.length - 1]
  }
  toArray(): readonly T[] {
    return this.items
  }

  push(item: T) {
    const size = this.sizeOf(item)
    this.items.push(item)
    this.sizes.push(size)
    this.bytes += size
    if (this.items.length > this.maxItems || this.bytes > this.maxBytes) this.thin()
  }

  clear() {
    this.items = []
    this.sizes = []
    this.bytes = 0
  }

  /** Keep every 2nd item in the oldest half (always keep index 0). */
  private thin() {
    const half = Math.floor(this.items.length / 2)
    const items: T[] = []
    const sizes: number[] = []
    for (let i = 0; i < this.items.length; i++) {
      if (i >= half || i % 2 === 0) {
        items.push(this.items[i])
        sizes.push(this.sizes[i])
      }
    }
    this.items = items
    this.sizes = sizes
    this.bytes = sizes.reduce((a, b) => a + b, 0)
  }

  /** Index of the item whose key is closest to `key` (items must be ordered by key). */
  nearest(key: number, keyOf: (t: T) => number): number {
    let lo = 0
    let hi = this.items.length - 1
    if (hi < 0) return -1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (keyOf(this.items[mid]) < key) lo = mid + 1
      else hi = mid
    }
    if (
      lo > 0 &&
      Math.abs(keyOf(this.items[lo - 1]) - key) <= Math.abs(keyOf(this.items[lo]) - key)
    )
      return lo - 1
    return lo
  }
}
