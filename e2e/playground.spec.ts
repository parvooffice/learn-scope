import { expect, test } from '@playwright/test'

test('playground trains, the boundary changes, and the timeline travels back', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })

  await page.goto('/#/playground?play=1')
  const num = async (id: string) =>
    Number((await page.getByTestId(id).innerText()).replace('%', ''))

  // wait for the first snapshot (model built, worker ready)
  await expect(page.getByTestId('stat-train-loss')).not.toHaveText('–')
  const lossBefore = await num('stat-train-loss')
  const boundary = page.getByRole('img', { name: /Decision boundary/ })
  const picBefore = await boundary.evaluate((c) => (c as HTMLCanvasElement).toDataURL())

  // training advances
  await expect.poll(() => num('stat-step'), { timeout: 90_000 }).toBeGreaterThan(150)
  const lossAfter = await num('stat-train-loss')
  expect(lossAfter).toBeLessThan(lossBefore)
  const picAfter = await boundary.evaluate((c) => (c as HTMLCanvasElement).toDataURL())
  expect(picAfter).not.toEqual(picBefore)

  // pause, then scrub the timeline back to the start
  await page.getByRole('button', { name: 'Pause training' }).click()
  const latestStep = await num('stat-step')
  const slider = page.getByLabel(/Timeline scrubber/)
  await slider.focus()
  await page.keyboard.press('Home')
  await expect.poll(() => num('stat-step')).toBeLessThan(latestStep)
  await expect(page.getByRole('button', { name: 'Back to live' })).toBeVisible()
  // and the loss shown at the start is higher than at the end: every view follows the selected step
  expect(await num('stat-train-loss')).toBeGreaterThan(lossAfter)

  expect(errors.filter((e) => !/WebGPU|powerPreference|GPU stall|requestAdapter/i.test(e))).toEqual(
    [],
  )
})

test('landing page loads and the Try it button opens the playground', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Watch a neural network')
  await page.getByRole('button', { name: /Try it/ }).click()
  await expect(page).toHaveURL(/#\/playground/)
  await expect(page.getByRole('button', { name: /Pause training|Start training/ })).toBeVisible()
})

test('backprop explainer walks through the steps with live numbers', async ({ page }) => {
  await page.goto('/#/backprop')
  await expect(page.getByText('0 · The network and its numbers')).toBeVisible()
  for (let i = 0; i < 8; i++) await page.getByRole('button', { name: 'Next step' }).click()
  await expect(
    page.getByText(/hand-written chain rule vs autograd engine: max difference = [0-9.]+e[+-]\d+/),
  ).toBeVisible()
})
