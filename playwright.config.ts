import { defineConfig } from '@playwright/test'

// Uses an installed Chrome/Edge by default so no browser download is needed.
// Set PW_CHANNEL=chromium to use Playwright's own bundled browser instead.
const channel = process.env.PW_CHANNEL ?? 'chrome'

export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  expect: { timeout: 60_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  use: {
    baseURL: 'http://localhost:4173',
    channel: channel === 'chromium' ? undefined : channel,
    headless: true,
    launchOptions: { args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'] },
  },
  webServer: {
    command: 'npm run build && npm run preview -- --strictPort',
    url: 'http://localhost:4173',
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
})
