import { navigate, useRoute, type Route } from '../lib/router'
import { useUI } from '../store/ui'

const LINKS: { route: Route; label: string }[] = [
  { route: 'playground', label: 'Playground' },
  { route: 'backprop', label: 'Backprop' },
  { route: 'mnist', label: 'MNIST' },
  { route: 'about', label: 'About' },
]

export function Header() {
  const { route } = useRoute()
  const theme = useUI((s) => s.theme)
  const toggle = useUI((s) => s.toggleTheme)
  return (
    <header className="flex h-12 flex-none items-center gap-3 border-b border-line bg-panel px-3">
      <a
        href="#/"
        className="flex items-center gap-2 text-sm font-semibold tracking-tight text-ink"
        aria-label="LearnScope home"
      >
        <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden>
          <rect width="32" height="32" rx="7" fill="var(--panel2)" stroke="var(--border)" />
          <path d="M5 22 C10 22 10 8 16 8 S22 22 27 22" fill="none" stroke="var(--accent)" strokeWidth="2.6" strokeLinecap="round" />
          <circle cx="16" cy="8" r="2.6" fill="var(--accent)" />
        </svg>
        Learn<span className="text-accent">Scope</span>
      </a>
      <nav className="ml-2 flex items-center gap-1" aria-label="Main">
        {LINKS.map((l) => (
          <button
            key={l.route}
            type="button"
            onClick={() => navigate(l.route, l.route === 'playground' ? 'play=0' : undefined)}
            aria-current={route === l.route ? 'page' : undefined}
            className={`rounded-md px-2.5 py-1 text-sm ${
              route === l.route ? 'bg-panel2 text-accent' : 'text-muted hover:text-ink'
            }`}
          >
            {l.label}
          </button>
        ))}
      </nav>
      <div className="ml-auto flex items-center gap-2">
        <span className="hidden text-[11px] text-muted md:inline">
          Space = play/pause · ← → = step / scrub · R = reset
        </span>
        <button
          type="button"
          onClick={toggle}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          className="rounded-md border border-line px-2 py-1 text-sm text-muted hover:text-ink"
        >
          {theme === 'dark' ? '☀' : '☾'}
        </button>
      </div>
    </header>
  )
}
