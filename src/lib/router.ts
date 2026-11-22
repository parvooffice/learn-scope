import { useSyncExternalStore } from 'react'

export type Route = 'home' | 'playground' | 'compare' | 'mnist' | 'backprop' | 'about'

function parse(): { route: Route; query: URLSearchParams } {
  const raw = window.location.hash.replace(/^#\/?/, '')
  const [path, qs] = raw.split('?')
  const r = (path || 'home') as Route
  const known: Route[] = ['home', 'playground', 'compare', 'mnist', 'backprop', 'about']
  return { route: known.includes(r) ? r : 'home', query: new URLSearchParams(qs ?? '') }
}

let cache = { hash: '\u0000', value: parse() }
function snapshot() {
  if (cache.hash !== window.location.hash) cache = { hash: window.location.hash, value: parse() }
  return cache.value
}
function subscribe(cb: () => void) {
  window.addEventListener('hashchange', cb)
  return () => window.removeEventListener('hashchange', cb)
}

export function useRoute() {
  return useSyncExternalStore(subscribe, snapshot, snapshot)
}

export function navigate(route: Route, query?: string) {
  window.location.hash = `/${route}${query ? `?${query}` : ''}`
}

/** Replace the query without adding history entries or re-triggering navigation loops. */
export function replaceQuery(route: Route, query: string) {
  const url = `${window.location.pathname}#/${route}${query ? `?${query}` : ''}`
  window.history.replaceState(null, '', url)
}
