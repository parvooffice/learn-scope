const STACK = [
  ['TypeScript (strict) + React 19 + Vite', 'the app'],
  ['Tailwind CSS v4 + CSS variables', 'styling and the dark/light themes'],
  ['TensorFlow.js (WebGPU → WebGL → CPU)', 'GPU training, in Web Workers'],
  ['Hand-written autograd (src/engine/tiny)', 'powers the backprop explainer'],
  [
    'Hand-written CPU MLP (src/engine/cpu)',
    'the landing-page animation and a gradient cross-check',
  ],
  ['Canvas 2D + d3-scale', 'all the charts, graphs and heat maps'],
  ['Zustand + idb', 'state, and IndexedDB caching of MNIST'],
  ['Vitest + Playwright', 'unit tests and an end-to-end test'],
]

export default function About() {
  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-3xl space-y-8 px-5 py-10">
        <header>
          <h1 className="text-3xl font-bold">
            About Learn<span className="text-accent">Scope</span>
          </h1>
          <p className="mt-3 text-muted">
            LearnScope lets you watch a neural network learn in real time. Instead of staring at one
            loss number you can see every weight, activation and gradient; pause and rewind to any
            training step; and let a rule-based “Model Doctor” point out what is going wrong.
            Everything runs in your browser: there is no server and no account.
          </p>
        </header>

        <section>
          <h2 className="mb-2 text-lg font-semibold">What to try</h2>
          <ol className="list-decimal space-y-1.5 pl-5 text-sm text-muted">
            <li>
              <b className="text-ink">Playground</b>: pick the spiral, press play, then drag the
              timeline back. Raise the learning rate to 1+ and watch the Doctor catch it, then click
              its fix.
            </li>
            <li>
              <b className="text-ink">Backprop</b>: step through one training step with real
              numbers.
            </li>
            <li>
              <b className="text-ink">MNIST</b>: train a CNN on handwritten digits, then draw your
              own digit.
            </li>
          </ol>
        </section>

        <section>
          <h2 className="mb-2 text-lg font-semibold">Built with</h2>
          <ul className="divide-y divide-line rounded-xl border border-line bg-panel text-sm">
            {STACK.map(([a, b]) => (
              <li key={a} className="flex justify-between gap-4 px-3 py-2">
                <span>{a}</span>
                <span className="text-right text-muted">{b}</span>
              </li>
            ))}
          </ul>
        </section>

        <section>
          <h2 className="mb-2 text-lg font-semibold">An honest note</h2>
          <p className="text-sm leading-relaxed text-muted">
            This project was built with AI assistance. The goal is learning: the autograd engine,
            the diagnostic rules and the math behind each visualisation are documented in the
            repository (<code>docs/LEARNING.md</code>, <code>docs/ARCHITECTURE.md</code>) so that
            every part can be explained, not just run. Performance numbers shown in the app (images
            per second, time per epoch) are measured live on your own device.
          </p>
        </section>

        <section>
          <h2 className="mb-2 text-lg font-semibold">Browser support</h2>
          <p className="text-sm leading-relaxed text-muted">
            Best in current Chrome or Edge, which have WebGPU. In Firefox and Safari the app
            automatically falls back to WebGL (still GPU) or CPU and shows a notice. Training is the
            same; it may just be slower.
          </p>
        </section>
      </div>
    </div>
  )
}
