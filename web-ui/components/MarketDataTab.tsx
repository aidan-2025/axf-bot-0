import { MarketData } from '../lib/api';

interface MarketDataTabProps {
  market: MarketData | null;
  selectedPair: string | null;
  onPairSelect: (pair: string) => void;
}

export default function MarketDataTab({ market, selectedPair, onPairSelect }: MarketDataTabProps) {
  return (
    <div className="space-y-6">
      {/* Market Data Table */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Live Market Data</h3>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-gray-500">Live</span>
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">24h Change</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">24h %</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Volume</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">High/Low</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {(Array.isArray(market) ? market : []).map((cp) => (
                  <tr key={cp.symbol} className="hover:bg-gray-50">
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="text-sm font-medium text-gray-900">{cp.symbol}</div>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">
                      {cp.currentPrice.toFixed(5)}
                    </td>
                    <td className={`px-4 py-4 whitespace-nowrap text-sm ${
                      cp.change24h >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {cp.change24h >= 0 ? '+' : ''}{cp.change24h.toFixed(5)}
                    </td>
                    <td className={`px-4 py-4 whitespace-nowrap text-sm ${
                      cp.changePercent24h >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {cp.changePercent24h >= 0 ? '+' : ''}{cp.changePercent24h.toFixed(2)}%
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">
                      {cp.volume.toLocaleString()}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">
                      {cp.high24h?.toFixed(5) || 'N/A'} / {cp.low24h?.toFixed(5) || 'N/A'}
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm font-medium">
                      <button 
                        onClick={() => onPairSelect(cp.symbol)}
                        className="text-blue-600 hover:text-blue-900"
                      >
                        Analyze
                      </button>
                    </td>
                  </tr>
                ))}
                {(!market || !Array.isArray(market) || market.length === 0) && (
                  <tr>
                    <td colSpan={7} className="px-4 py-6 text-center text-sm text-gray-500">
                      No market data available
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
