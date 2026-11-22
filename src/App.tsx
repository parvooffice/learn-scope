import { lazy, Suspense } from 'react'
import { Header } from './components/Header'
import { useRoute } from './lib/router'
import Landing from './app/Landing'

// Heavy pages (TensorFlow.js, charts) are code-split so the landing page stays tiny.
const Playground = lazy(() => import('./app/playground/Playground'))
const Backprop = lazy(() => import('./app/backprop/Backprop'))
const Mnist = lazy(() => import('./app/mnist/Mnist'))
const About = lazy(() => import('./app/About'))

function Loading() {
  return (
    <div className="flex h-full items-center justify-center text-sm text-muted" role="status">
      Loading…
    </div>
  )
}

export default function App() {
  const { route } = useRoute()
  return (
    <div className="flex h-full flex-col">
      <Header />
      <div className="min-h-0 flex-1">
        <Suspense fallback={<Loading />}>
          {route === 'home' && <Landing />}
          {route === 'playground' && <Playground />}
          {route === 'backprop' && <Backprop />}
          {route === 'mnist' && <Mnist />}
          {route === 'about' && <About />}
        </Suspense>
      </div>
    </div>
  )
}
