import { navigate } from '../lib/router'

export default function Landing() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
      <h1 className="text-4xl font-bold">Learn<span className="text-accent">Scope</span></h1>
      <button type="button" onClick={() => navigate('playground')} className="rounded-lg bg-accent px-4 py-2 font-medium text-accent-ink">
        Try it
      </button>
    </div>
  )
}
