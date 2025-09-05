import { useState, useEffect } from 'react';
import Head from 'next/head';
import { MarketData } from '../lib/api';

interface SystemHealth {
  status: string;
  timestamp: string;
}

export default function WorkingDashboard() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [market, setMarket] = useState<MarketData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const fetchData = async () => {
      try {
        console.log('Fetching data...');
        
        // Fetch system health
        const healthResponse = await fetch('http://localhost:8000/health');
        if (!healthResponse.ok) throw new Error(`Health API failed: ${healthResponse.status}`);
        const healthData = await healthResponse.json();
        setHealth(healthData);
        console.log('Health data loaded:', healthData);

        // Fetch market data
        const marketResponse = await fetch('http://localhost:8000/api/v1/data/market');
        if (!marketResponse.ok) throw new Error(`Market API failed: ${marketResponse.status}`);
        const marketData = await marketResponse.json();
        setMarket(marketData);
        console.log('Market data loaded:', marketData);

      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [mounted]);

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Initializing...</p>
        </div>
      </div>
    );
  }

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

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 mb-4">Dashboard Error</h1>
          <p className="text-gray-600 mb-4">Error: {error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>AXF Bot 0 - Working Dashboard</title>
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

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
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
                        <dt className="text-sm font-medium text-gray-500 truncate">System Status</dt>
                        <dd className="text-lg font-medium text-gray-900 capitalize">{health?.status || 'Unknown'}</dd>
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
                        <dt className="text-sm font-medium text-gray-500 truncate">Currency Pairs</dt>
                        <dd className="text-lg font-medium text-gray-900">
                          {market?.length || 0}
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
                        <span className="text-white font-bold">📊</span>
                      </div>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">Data Source</dt>
                        <dd className="text-lg font-medium text-gray-900">Live API</dd>
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
                        <span className="text-white font-bold">🤖</span>
                      </div>
                    </div>
                    <div className="ml-5 w-0 flex-1">
                      <dl>
                        <dt className="text-sm font-medium text-gray-500 truncate">AI Status</dt>
                        <dd className="text-lg font-medium text-gray-900">Ready</dd>
                      </dl>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Live Market Data */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg leading-6 font-medium text-gray-900">Live Market Data</h3>
                  <span className="text-xs text-gray-500">Real-time from App1 API</span>
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
                          <td className={`px-4 py-2 whitespace-nowrap ${cp.change24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {cp.change24h.toFixed(5)}
                          </td>
                          <td className={`px-4 py-2 whitespace-nowrap ${cp.changePercent24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {cp.changePercent24h.toFixed(2)}%
                          </td>
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

            {/* AI Features Placeholder */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">AI-Powered Features</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="text-center py-8">
                    <div className="text-gray-400 mb-4">
                      <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                      </svg>
                    </div>
                    <h4 className="text-lg font-medium text-gray-900 mb-2">Strategy Generation</h4>
                    <p className="text-gray-600 mb-4">AI-powered forex trading strategy generation using real-time market analysis.</p>
                    <button className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors">
                      Generate Strategy
                    </button>
                  </div>
                  
                  <div className="text-center py-8">
                    <div className="text-gray-400 mb-4">
                      <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <h4 className="text-lg font-medium text-gray-900 mb-2">Market Analysis</h4>
                    <p className="text-gray-600 mb-4">Comprehensive market analysis with sentiment, news, and technical indicators.</p>
                    <button className="bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 transition-colors">
                      Analyze Market
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
