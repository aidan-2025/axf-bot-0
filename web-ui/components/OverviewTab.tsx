import { MarketData } from '../lib/api';

interface OverviewTabProps {
  market: MarketData | null;
}

export default function OverviewTab({ market }: OverviewTabProps) {
  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-green-500 rounded-md flex items-center justify-center">
                  <span className="text-white font-bold">$</span>
                </div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Total Pairs</dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {Array.isArray(market) ? market.length : 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                  <span className="text-white font-bold">📊</span>
                </div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Active Strategies</dt>
                  <dd className="text-lg font-medium text-gray-900">3</dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-purple-500 rounded-md flex items-center justify-center">
                  <span className="text-white font-bold">💰</span>
                </div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">Portfolio Value</dt>
                  <dd className="text-lg font-medium text-gray-900">$12,450.00</dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-yellow-500 rounded-md flex items-center justify-center">
                  <span className="text-white font-bold">📈</span>
                </div>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">24h P&L</dt>
                  <dd className="text-lg font-medium text-green-600">+$245.30</dd>
                </dl>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Quick Actions</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <button className="bg-blue-600 text-white px-4 py-3 rounded-md hover:bg-blue-700 transition-colors text-left">
              <div className="flex items-center">
                <span className="text-2xl mr-3">🤖</span>
                <div>
                  <div className="font-medium">Generate Strategy</div>
                  <div className="text-sm opacity-90">AI-powered strategy creation</div>
                </div>
              </div>
            </button>
            
            <button className="bg-green-600 text-white px-4 py-3 rounded-md hover:bg-green-700 transition-colors text-left">
              <div className="flex items-center">
                <span className="text-2xl mr-3">📊</span>
                <div>
                  <div className="font-medium">Market Analysis</div>
                  <div className="text-sm opacity-90">Real-time market insights</div>
                </div>
              </div>
            </button>
            
            <button className="bg-purple-600 text-white px-4 py-3 rounded-md hover:bg-purple-700 transition-colors text-left">
              <div className="flex items-center">
                <span className="text-2xl mr-3">⚙️</span>
                <div>
                  <div className="font-medium">Configure Bot</div>
                  <div className="text-sm opacity-90">Adjust trading parameters</div>
                </div>
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
