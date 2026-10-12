import { createFileRoute } from '@tanstack/react-router'
import { getStats } from '../lib/store'

export const Route = createFileRoute('/analytics')({
  component: Analytics,
})

function Analytics() {
  const stats = getStats()

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Analytics Dashboard</h1>
      <div className="grid grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Total Inspections</div>
          <div className="text-3xl font-bold mt-2">{stats.total}</div>
        </div>
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Defect Rate</div>
          <div className="text-3xl font-bold mt-2">{stats.defectRate}%</div>
        </div>
        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
          <div className="text-sm text-gray-500 font-medium">Pass Rate</div>
          <div className="text-3xl font-bold mt-2">{100 - stats.defectRate}%</div>
        </div>
      </div>
      
      <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm mt-8">
        <h3 className="font-bold mb-4">Model Performance Overview</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex justify-between p-3 border-b"><span className="text-gray-500">Accuracy</span><span className="font-medium">99.63%</span></div>
          <div className="flex justify-between p-3 border-b"><span className="text-gray-500">Macro F1</span><span className="font-medium">99.63%</span></div>
          <div className="flex justify-between p-3 border-b"><span className="text-gray-500">Avg Inference</span><span className="font-medium">0.55 ms</span></div>
          <div className="flex justify-between p-3 border-b"><span className="text-gray-500">Parameters</span><span className="font-medium">4.02M</span></div>
        </div>
      </div>
    </div>
  )
}
