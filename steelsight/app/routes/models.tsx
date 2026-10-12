import { createFileRoute } from '@tanstack/react-router'

export const Route = createFileRoute('/models')({
  component: Models,
})

function Models() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Models</h1>
      
      <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold">EfficientNet-B0 (Deployed)</h2>
          <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-bold rounded">ACTIVE</span>
        </div>
        <p className="text-gray-600 text-sm mb-6">Lightweight CNN architecture optimized for edge deployment and fast inference on industrial lines.</p>
        
        <table className="w-full text-left text-sm">
          <thead className="bg-gray-50">
            <tr><th className="p-3">Model</th><th className="p-3">Accuracy</th><th className="p-3">Latency</th><th className="p-3">Size</th></tr>
          </thead>
          <tbody className="divide-y">
            <tr><td className="p-3 font-medium">Baseline CNN</td><td className="p-3">89.2%</td><td className="p-3">12ms</td><td className="p-3">1.2MB</td></tr>
            <tr><td className="p-3 font-medium">ResNet-50</td><td className="p-3">97.1%</td><td className="p-3">140ms</td><td className="p-3">98MB</td></tr>
            <tr className="bg-blue-50"><td className="p-3 font-medium text-blue-900">EfficientNet-B0</td><td className="p-3 font-medium text-blue-900">96.8%</td><td className="p-3 font-medium text-blue-900">42ms</td><td className="p-3 font-medium text-blue-900">21MB</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
