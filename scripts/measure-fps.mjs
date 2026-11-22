// Measures UI frame rate and long main-thread tasks while the playground trains.
// usage: npm run build && npm run preview   (in another terminal), then: npm run perf
import { chromium } from '@playwright/test'

const url = process.env.URL ?? 'http://localhost:4173/#/playground?play=1'
const browser = await chromium.launch({
  channel: process.env.PW_CHANNEL === 'chromium' ? undefined : (process.env.PW_CHANNEL ?? 'chrome'),
  headless: true,
  args: ['--enable-unsafe-webgpu'],
})
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } })
await page.goto(url)
await page.waitForFunction(
  () => Number(document.querySelector('[data-testid=stat-step]')?.textContent) > 50,
  null,
  {
    timeout: 90_000,
  },
)

const result = await page.evaluate(
  (ms) =>
    new Promise((resolve) => {
      const frames = []
      let longTasks = 0
      let longest = 0
      try {
        new PerformanceObserver((list) => {
          for (const e of list.getEntries()) {
            longTasks++
            longest = Math.max(longest, e.duration)
          }
        }).observe({ entryTypes: ['longtask'] })
      } catch {
        /* longtask not supported */
      }
      const step0 = Number(document.querySelector('[data-testid=stat-step]').textContent)
      let last = performance.now()
      const t0 = last
      const tick = (t) => {
        frames.push(t - last)
        last = t
        if (t - t0 < ms) requestAnimationFrame(tick)
        else {
          const sorted = [...frames].sort((a, b) => a - b)
          resolve({
            seconds: (t - t0) / 1000,
            fps: frames.length / ((t - t0) / 1000),
            p95FrameMs: sorted[Math.floor(sorted.length * 0.95)],
            worstFrameMs: sorted[sorted.length - 1],
            longTasks,
            longestTaskMs: longest,
            stepsTrained:
              Number(document.querySelector('[data-testid=stat-step]').textContent) - step0,
            backend: document.body.innerText.match(/GPU · \w+|CPU fallback/)?.[0] ?? 'unknown',
          })
        }
      }
      requestAnimationFrame(tick)
    }),
  8000,
)
console.log(JSON.stringify(result, null, 2))
await browser.close()
