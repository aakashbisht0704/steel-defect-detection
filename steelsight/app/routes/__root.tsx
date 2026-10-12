import { createRootRoute, Outlet } from '@tanstack/react-router'
import { Navbar } from '../components/layout/Navbar'

export const Route = createRootRoute({
  component: () => (
    <div className="min-h-screen font-sans flex flex-col">
      <Navbar />
      <main className="flex-1 p-6 max-w-6xl w-full mx-auto">
        <Outlet />
      </main>
    </div>
  ),
})
