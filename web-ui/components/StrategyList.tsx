import { useState } from 'react';
import { 
  EyeIcon, 
  PlayIcon, 
  PauseIcon, 
  TrashIcon,
  ChartBarIcon,
  ClockIcon,
  CurrencyDollarIcon,
  TrendingUpIcon,
  TrendingDownIcon
} from '@heroicons/react/24/outline';

interface Strategy {
  id: string;
  name: string;
  description: string;
  strategyType: string;
  currencyPairs: string[];
  timeframes: string[];
  status: 'active' | 'paused' | 'archived' | 'pending';
  performance: {
    profitFactor: number;
    winRate: number;
    maxDrawdown: number;
    sharpeRatio: number;
    totalTrades: number;
    totalProfit: number;
  };
  createdAt: string;
  lastUpdated: string;
}

interface StrategyListProps {
  strategies: Strategy[];
}

export default function StrategyList({ strategies }: StrategyListProps) {
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('createdAt');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const filteredStrategies = strategies.filter(strategy => {
    if (filter === 'all') return true;
    return strategy.status === filter;
  });

  const sortedStrategies = [...filteredStrategies].sort((a, b) => {
    let aValue, bValue;
    
    switch (sortBy) {
      case 'name':
        aValue = a.name.toLowerCase();
        bValue = b.name.toLowerCase();
        break;
      case 'profitFactor':
        aValue = a.performance.profitFactor;
        bValue = b.performance.profitFactor;
        break;
      case 'winRate':
        aValue = a.performance.winRate;
        bValue = b.performance.winRate;
        break;
      case 'totalProfit':
        aValue = a.performance.totalProfit;
        bValue = b.performance.totalProfit;
        break;
      case 'createdAt':
      default:
        aValue = new Date(a.createdAt).getTime();
        bValue = new Date(b.createdAt).getTime();
        break;
    }

    if (sortOrder === 'asc') {
      return aValue > bValue ? 1 : -1;
    } else {
      return aValue < bValue ? 1 : -1;
    }
  });

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <span className="badge badge-success">Active</span>;
      case 'paused':
        return <span className="badge badge-warning">Paused</span>;
      case 'archived':
        return <span className="badge badge-danger">Archived</span>;
      case 'pending':
        return <span className="badge badge-info">Pending</span>;
      default:
        return <span className="badge">Unknown</span>;
    }
  };

  const getPerformanceColor = (value: number, type: 'profit' | 'rate' | 'drawdown') => {
    if (type === 'drawdown') {
      return value > 15 ? 'text-danger-600' : value > 10 ? 'text-warning-600' : 'text-success-600';
    }
    return value > 0 ? 'text-success-600' : 'text-danger-600';
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(amount);
  };

  const formatPercent = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Filters and Controls */}
      <div className="card">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-4 sm:space-y-0">
          <div className="flex items-center space-x-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Filter by Status</label>
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="select"
              >
                <option value="all">All Strategies</option>
                <option value="active">Active</option>
                <option value="paused">Paused</option>
                <option value="archived">Archived</option>
                <option value="pending">Pending</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Sort by</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="select"
              >
                <option value="createdAt">Created Date</option>
                <option value="name">Name</option>
                <option value="profitFactor">Profit Factor</option>
                <option value="winRate">Win Rate</option>
                <option value="totalProfit">Total Profit</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Order</label>
              <select
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                className="select"
              >
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </div>
          </div>
          <div className="text-sm text-gray-500">
            {sortedStrategies.length} of {strategies.length} strategies
          </div>
        </div>
      </div>

      {/* Strategy Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {sortedStrategies.map((strategy) => (
          <div key={strategy.id} className="card hover:shadow-md transition-shadow">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-gray-900 mb-1">
                  {strategy.name}
                </h3>
                <p className="text-sm text-gray-600 mb-2">
                  {strategy.description}
                </p>
                <div className="flex items-center space-x-2">
                  {getStatusBadge(strategy.status)}
                  <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
                    {strategy.strategyType.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
              </div>
              <div className="flex items-center space-x-1">
                <button className="p-1 text-gray-400 hover:text-gray-600">
                  <EyeIcon className="h-5 w-5" />
                </button>
                {strategy.status === 'active' ? (
                  <button className="p-1 text-warning-400 hover:text-warning-600">
                    <PauseIcon className="h-5 w-5" />
                  </button>
                ) : (
                  <button className="p-1 text-success-400 hover:text-success-600">
                    <PlayIcon className="h-5 w-5" />
                  </button>
                )}
                <button className="p-1 text-danger-400 hover:text-danger-600">
                  <TrashIcon className="h-5 w-5" />
                </button>
              </div>
            </div>

            {/* Currency Pairs and Timeframes */}
            <div className="mb-4">
              <div className="flex flex-wrap gap-1 mb-2">
                {strategy.currencyPairs.slice(0, 3).map((pair) => (
                  <span
                    key={pair}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                  >
                    {pair}
                  </span>
                ))}
                {strategy.currencyPairs.length > 3 && (
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                    +{strategy.currencyPairs.length - 3} more
                  </span>
                )}
              </div>
              <div className="flex flex-wrap gap-1">
                {strategy.timeframes.map((timeframe) => (
                  <span
                    key={timeframe}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                  >
                    {timeframe}
                  </span>
                ))}
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="text-center">
                <div className={`text-lg font-semibold ${getPerformanceColor(strategy.performance.profitFactor, 'profit')}`}>
                  {strategy.performance.profitFactor.toFixed(2)}
                </div>
                <div className="text-xs text-gray-500">Profit Factor</div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-semibold ${getPerformanceColor(strategy.performance.winRate, 'rate')}`}>
                  {formatPercent(strategy.performance.winRate)}
                </div>
                <div className="text-xs text-gray-500">Win Rate</div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-semibold ${getPerformanceColor(strategy.performance.maxDrawdown, 'drawdown')}`}>
                  {formatPercent(strategy.performance.maxDrawdown)}
                </div>
                <div className="text-xs text-gray-500">Max DD</div>
              </div>
              <div className="text-center">
                <div className={`text-lg font-semibold ${getPerformanceColor(strategy.performance.sharpeRatio, 'profit')}`}>
                  {strategy.performance.sharpeRatio.toFixed(2)}
                </div>
                <div className="text-xs text-gray-500">Sharpe</div>
              </div>
            </div>

            {/* Total Profit and Trades */}
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center space-x-1">
                <CurrencyDollarIcon className="h-4 w-4 text-gray-400" />
                <span className={`font-medium ${getPerformanceColor(strategy.performance.totalProfit, 'profit')}`}>
                  {formatCurrency(strategy.performance.totalProfit)}
                </span>
              </div>
              <div className="flex items-center space-x-1">
                <ChartBarIcon className="h-4 w-4 text-gray-400" />
                <span className="text-gray-600">
                  {strategy.performance.totalTrades} trades
                </span>
              </div>
            </div>

            {/* Timestamps */}
            <div className="mt-4 pt-4 border-t border-gray-200">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <div className="flex items-center space-x-1">
                  <ClockIcon className="h-3 w-3" />
                  <span>Created {new Date(strategy.createdAt).toLocaleDateString()}</span>
                </div>
                <span>Updated {new Date(strategy.lastUpdated).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {sortedStrategies.length === 0 && (
        <div className="text-center py-12">
          <ChartBarIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-medium text-gray-900">No strategies found</h3>
          <p className="mt-1 text-sm text-gray-500">
            {filter === 'all' 
              ? 'No strategies have been generated yet.' 
              : `No ${filter} strategies found.`
            }
          </p>
        </div>
      )}
    </div>
  );
}
