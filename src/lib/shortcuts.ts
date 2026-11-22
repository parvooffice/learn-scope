import { useEffect } from 'react'

export interface Shortcut {
  key: string
  shift?: boolean
  run: () => void
}

/** Global keyboard shortcuts that don't fire while typing in form fields. */
export function useShortcuts(list: Shortcut[]) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      const tag = t?.tagName
      if (tag === 'INPUT' && (t as HTMLInputElement).type !== 'range') return
      if (tag === 'TEXTAREA' || tag === 'SELECT' || t?.isContentEditable) return
      if (e.ctrlKey || e.metaKey || e.altKey) return
      const hit = list.find((s) => s.key === e.key && !!s.shift === e.shiftKey)
      if (hit) {
        e.preventDefault()
        hit.run()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })
}
