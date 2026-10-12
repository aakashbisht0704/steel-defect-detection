import { createFileRoute, Link } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { getHealth } from '../lib/ai-client'
import { getStats, getInspections } from '../lib/store'
import { Upload } from 'lucide-react'

export const Route = createFileRoute('/')({
  component: Index,
})

function Index() {
  const [status, setStatus] = useState('checking...')
  const stats = getStats()

  useEffect(() => {
    getHealth().then(res => setStatus(res.status === 'online' ? 'ONLINE' : 'OFFLINE'))
  }, [])

  return (
    <div className="space-y-12 mt-8">
      <div className="text-center space-y-4">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gray-100 text-xs font-semibold">
          <span className={`w-2 h-2 rounded-full ${status === 'ONLINE' ? 'bg-green-500' : 'bg-red-500'}`}></span>
          AI ENGINE {status}
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight">AI SURFACE INSPECTION</h1>
        <p className="text-gray-500 text-lg">Detect and classify steel surface defects in milliseconds.</p>
      </div>
      
      <Link to="/inspect" className="block max-w-2xl mx-auto border-2 border-dashed border-gray-300 rounded-xl p-12 text-center hover:bg-gray-50 transition-colors bg-white">
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <div className="text-lg font-medium">START INSPECTION</div>
        <div className="text-sm text-gray-500">Drag and drop or click to upload</div>
      </Link>

      <div className="grid grid-cols-3 gap-6 max-w-4xl mx-auto">
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Total Inspections</div>
          <div className="text-3xl font-bold mt-2">{stats.total}</div>
        </div>
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Defects Found</div>
          <div className="text-3xl font-bold mt-2 text-red-600">{stats.defects}</div>
        </div>
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Passed</div>
          <div className="text-3xl font-bold mt-2 text-green-600">{stats.passed}</div>
        </div>
      </div>
    </div>
  )
}
