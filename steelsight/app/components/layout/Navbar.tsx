import { Link } from '@tanstack/react-router';

export function Navbar() {
  return (
    <nav className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
      <div className="flex items-center gap-6">
        <div className="font-bold text-xl tracking-tight">STEELSIGHT</div>
        <div className="flex gap-4 text-sm font-medium text-gray-600">
          <Link to="/" className="hover:text-black [&.active]:text-black">Dashboard</Link>
          <Link to="/inspect" className="hover:text-black [&.active]:text-black">Inspect</Link>
          <Link to="/history" className="hover:text-black [&.active]:text-black">History</Link>
          <Link to="/analytics" className="hover:text-black [&.active]:text-black">Analytics</Link>
          <Link to="/models" className="hover:text-black [&.active]:text-black">Models</Link>
        </div>
      </div>
      <div>
        <Link to="/inspect" className="bg-[#1a56db] text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-blue-700">
          + New Inspection
        </Link>
      </div>
    </nav>
  );
}
