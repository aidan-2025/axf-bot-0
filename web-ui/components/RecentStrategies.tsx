import { Strategy } from '../lib/api';

interface RecentStrategiesProps {
  strategies: Strategy[];
}

export default function RecentStrategies({ strategies }: RecentStrategiesProps) {
  return (
    <div className="card">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Strategies</h3>
      <div className="space-y-3">
        {strategies.length > 0 ? (
          strategies.map((strategy) => (
            <div key={strategy.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <h4 className="font-medium text-gray-900">{strategy.name}</h4>
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    strategy.status === 'active' 
                      ? 'bg-success-100 text-success-800'
                      : strategy.status === 'paused'
                      ? 'bg-warning-100 text-warning-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {strategy.status}
                  </span>
                </div>
                <p className="text-sm text-gray-500 mt-1">
                  {strategy.currencyPairs.join(', ')} • {strategy.timeframes.join(', ')}
                </p>
              </div>
              <div className="text-right">
                <div className={`text-sm font-medium ${
                  strategy.performance.totalProfit >= 0 ? 'text-success-600' : 'text-danger-600'
                }`}>
                  ${strategy.performance.totalProfit.toFixed(2)}
                </div>
                <div className="text-xs text-gray-500">
                  {strategy.performance.winRate.toFixed(1)}% win rate
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-6">
            <p className="text-gray-500">No recent strategies</p>
          </div>
        )}
      </div>
    </div>
  );
}
