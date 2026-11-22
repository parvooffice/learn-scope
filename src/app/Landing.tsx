import { navigate } from '../lib/router'
import { Hero } from './Hero'

const FEATURES = [
  {
    icon: '⏪',
    title: 'Time travel',
    body: 'Every training step is recorded. Drag the timeline back to any moment and the whole network (weights, gradients, activations) redraws exactly as it was.',
  },
  {
    icon: '🩺',
    title: 'Model Doctor',
    body: 'Deterministic rules read the telemetry and diagnose vanishing gradients, dead ReLUs, overfitting and bad learning rates. Each comes with evidence and a one-click fix.',
  },
  {
    icon: '∂',
    title: 'Backprop, visualized',
    body: 'Walk through the forward pass, the loss and the chain rule applied backward, with real numbers from a from-scratch autograd engine.',
  },
  {
    icon: '⚡',
    title: 'Real GPU training',
    body: 'TensorFlow.js on WebGPU (falling back to WebGL, then CPU) runs in a Web Worker, so the interface stays smooth. Includes a CNN on MNIST with a draw-a-digit pad.',
  },
]

export default function Landing() {
  return (
    <div className="h-full overflow-y-auto scroll-thin">
      <div className="mx-auto max-w-6xl px-5 py-10 md:py-16">
        <section className="grid items-center gap-10 md:grid-cols-[1.1fr_1fr]">
          <div>
            <p className="mb-3 inline-block rounded-full border border-line px-3 py-1 text-xs text-muted">
              runs 100% in your browser · no install · no account
            </p>
            <h1 className="text-4xl font-bold leading-[1.08] tracking-tight md:text-5xl">
              Watch a neural network <span className="text-accent">learn</span>, live.
            </h1>
            <p className="mt-4 max-w-xl text-lg text-muted">
              See every weight, activation and gradient while a model trains. Pause, rewind to any
              step, and let the Model Doctor tell you what's going wrong, instead of staring at a
              single loss number.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => navigate('playground', 'play=1')}
                className="rounded-xl bg-accent px-6 py-3 text-base font-semibold text-accent-ink shadow-lg transition hover:brightness-110"
              >
                Try it →
              </button>
              <button
                type="button"
                onClick={() => navigate('backprop')}
                className="rounded-xl border border-line bg-panel px-5 py-3 text-base font-medium text-ink hover:border-accent"
              >
                See backprop step by step
              </button>
              <button
                type="button"
                onClick={() => navigate('mnist')}
                className="rounded-xl border border-line bg-panel px-5 py-3 text-base font-medium text-ink hover:border-accent"
              >
                MNIST + draw a digit
              </button>
            </div>
          </div>
          <Hero />
        </section>

        <section className="mt-16 grid gap-4 sm:grid-cols-2 lg:grid-cols-4" aria-label="Features">
          {FEATURES.map((f) => (
            <article key={f.title} className="rounded-2xl border border-line bg-panel p-4">
              <div
                className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-panel2 text-lg text-accent"
                aria-hidden
              >
                {f.icon}
              </div>
              <h2 className="mb-1 font-semibold">{f.title}</h2>
              <p className="text-sm leading-relaxed text-muted">{f.body}</p>
            </article>
          ))}
        </section>

        <p className="mt-12 text-center text-xs text-muted">
          LearnScope is a learning project: TypeScript, React, TensorFlow.js, a hand-written
          autograd engine and a rule-based Model Doctor.
        </p>
      </div>
    </div>
  )
}
