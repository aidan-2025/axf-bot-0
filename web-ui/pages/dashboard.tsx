import { useState, useEffect } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import { apiService, type MarketData } from '../lib/api';

// Dynamically import components to avoid SSR issues
const AIStrategyGenerator = dynamic(() => import('../components/AIStrategyGenerator'), {
  ssr: false,
  loading: () => <div className="text-center py-8">Loading AI Strategy Generator...</div>
});

const MarketAnalysisDashboard = dynamic(() => import('../components/MarketAnalysisDashboard'), {
  ssr: false,
  loading: () => <div className="text-center py-8">Loading Market Analysis...</div>
});

interface SystemHealth {
  status: string;
  timestamp: string;
}

interface Strategy {
  id: string;
  name: string;
  description: string;
  status: string;
  performance: {
    profit_factor: number;
    win_rate: number;
    max_drawdown: number;
    total_profit: number;
  };
}

export default function Dashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [market, setMarket] = useState<MarketData[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'ai-generator' | 'market-analysis'>('overview');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const fetchData = async () => {
      try {
        // Fetch system health
        const healthResponse = await fetch('http://localhost:8000/health');
        const healthData = await healthResponse.json();
        setHealth(healthData);

        // Fetch strategies
        const strategiesResponse = await fetch('http://localhost:8000/api/v1/strategies/');
        const strategiesData = await strategiesResponse.json();
        setStrategies(strategiesData.data?.all || []);

        // Fetch market data (free source-backed endpoint)
        const marketData = await apiService.getMarketData();
        setMarket(marketData);
      } catch (error) {
        console.error('Error fetching data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [mounted]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>AXF Bot 0 - Dashboard</title>
        <meta name="description" content="AI-Powered Forex Trading System" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">AXF Bot 0</h1>
                <p className="text-gray-600">AI-Powered Forex Trading System</p>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-3 h-3 rounded-full ${
                    health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  <span className="text-sm text-gray-600">
                    {health?.status === 'healthy' ? 'System Healthy' : 'System Issues'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('overview')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'overview'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Overview
              </button>
              <button
                onClick={() => setActiveTab('ai-generator')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'ai-generator'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                AI Strategy Generator
              </button>
              <button
                onClick={() => setActiveTab('market-analysis')}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'market-analysis'
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                Market Analysis
              </button>
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {activeTab === 'overview' && (
            <div className="space-y-6">
            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white overflow-hidden shadow rounded-lg">
                <div className="p-5">
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <div className="w-8 h-8 bg-blue-500 rounded-md flex items-center justify-center">
                        <span className="text-white font-bold">S</span>
                      </div>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">Active Strategies</dt>
                        <dd className="text-lg font-medium text-gray-900">{strategies.length}</dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>

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
                        <dt className="text-sm font-medium text-gray-500 truncate">Total Profit</dt>
                        <dd className="text-lg font-medium text-gray-900">
                          ${strategies.reduce((sum, s) => sum + s.performance.total_profit, 0).toFixed(2)}
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
                      <div className="w-8 h-8 bg-yellow-500 rounded-md flex items-center justify-center">
                        <span className="text-white font-bold">%</span>
                      </div>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">Avg Win Rate</dt>
                        <dd className="text-lg font-medium text-gray-900">
                          {strategies.length > 0 
                            ? (strategies.reduce((sum, s) => sum + s.performance.win_rate, 0) / strategies.length).toFixed(1)
                            : '0.0'
                          }%
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
                      <div className="w-8 h-8 bg-red-500 rounded-md flex items-center justify-center">
                        <span className="text-white font-bold">!</span>
                      </div>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">System Status</dt>
                        <dd className="text-lg font-medium text-gray-900 capitalize">{health?.status || 'Unknown'}</dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Market (free data source) */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Live Market (Free Source)</h3>
                  <span className="text-xs text-gray-500">/data/market</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">24h Change</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">24h %</th>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Volume</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {(market || []).map((cp) => (
                        <tr key={cp.symbol}>
                          <td className="px-4 py-2 whitespace-nowrap font-medium text-gray-900">{cp.symbol}</td>
                          <td className="px-4 py-2 whitespace-nowrap">{cp.currentPrice.toFixed(5)}</td>
                          <td className={`px-4 py-2 whitespace-nowrap ${cp.change24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>{cp.change24h.toFixed(5)}</td>
                          <td className={`px-4 py-2 whitespace-nowrap ${cp.changePercent24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>{cp.changePercent24h.toFixed(2)}%</td>
                          <td className="px-4 py-2 whitespace-nowrap">{cp.volume.toLocaleString()}</td>
                        </tr>
                      ))}
                      {(!market || market.length === 0) && (
                        <tr>
                          <td colSpan={5} className="px-4 py-6 text-center text-sm text-gray-500">No market data available</td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Strategies List */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Active Strategies</h3>
                <div className="space-y-4">
                  {strategies.map((strategy) => (
                    <div key={strategy.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="text-lg font-medium text-gray-900">{strategy.name}</h4>
                          <p className="text-sm text-gray-600">{strategy.description}</p>
                          <div className="mt-2 flex space-x-4 text-sm text-gray-500">
                            <span>Status: <span className="font-medium capitalize">{strategy.status}</span></span>
                            <span>Profit Factor: <span className="font-medium">{strategy.performance.profit_factor}</span></span>
                            <span>Win Rate: <span className="font-medium">{strategy.performance.win_rate}%</span></span>
                            <span>Total Profit: <span className="font-medium">${strategy.performance.total_profit}</span></span>
                          </div>
                        </div>
                        <div className={`px-2 py-1 text-xs font-medium rounded-full ${
                          strategy.status === 'active' 
                            ? 'bg-green-100 text-green-800' 
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {strategy.status}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          )}

          {activeTab === 'ai-generator' && (
            <AIStrategyGenerator 
              onStrategyGenerated={(strategy) => {
                console.log('Strategy generated:', strategy);
                // You could add logic here to refresh the strategies list
              }}
            />
          )}

          {activeTab === 'market-analysis' && (
            <MarketAnalysisDashboard />
          )}
        </div>
      </div>
    </>
  );
}
