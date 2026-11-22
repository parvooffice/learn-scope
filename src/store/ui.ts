import { create } from 'zustand'
import type { ChartId } from '../doctor/types'

type Theme = 'dark' | 'light'

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem('ls-theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* storage may be blocked */
  }
  return 'dark'
}

interface UIState {
  theme: Theme
  toggleTheme: () => void
  /** a chart the Model Doctor wants us to look at; `nonce` re-triggers the flash animation */
  highlight: { chart: ChartId; nonce: number } | null
  flash: (chart: ChartId) => void
}

export const useUI = create<UIState>((set, get) => ({
  theme: initialTheme(),
  toggleTheme: () => {
    const next: Theme = get().theme === 'dark' ? 'light' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    try {
      localStorage.setItem('ls-theme', next)
    } catch {
      /* ignore */
    }
    set({ theme: next })
  },
  highlight: null,
  flash: (chart) => set({ highlight: { chart, nonce: Date.now() } }),
}))
