import { createFileRoute } from '@tanstack/react-router'
import { getInspections } from '../lib/store'

export const Route = createFileRoute('/history')({
  component: History,
})

function History() {
  const inspections = getInspections()

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
      <table className="w-full text-left text-sm">
        <thead className="bg-gray-50 border-b border-gray-200 text-gray-600 font-medium">
          <tr>
            <th className="p-4">ID</th>
            <th className="p-4">Defect</th>
            <th className="p-4">Confidence</th>
            <th className="p-4">Status</th>
            <th className="p-4">Date</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {inspections.map(i => (
            <tr key={i.id} className="hover:bg-gray-50 cursor-pointer">
              <td className="p-4 font-medium">{i.id}</td>
              <td className="p-4 capitalize">{i.prediction.class.replace(/_/g, ' ')}</td>
              <td className="p-4">{(i.prediction.confidence * 100).toFixed(1)}%</td>
              <td className="p-4">
                <span className={`px-2 py-1 rounded text-xs font-semibold ${i.status === 'PASS' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                  {i.status}
                </span>
              </td>
              <td className="p-4 text-gray-500">{new Date(i.createdAt).toLocaleString()}</td>
            </tr>
          ))}
          {inspections.length === 0 && (
            <tr>
              <td colSpan={5} className="p-8 text-center text-gray-500">No inspections yet.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
