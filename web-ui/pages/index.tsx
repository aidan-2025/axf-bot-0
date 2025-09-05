import Head from 'next/head';
import { useState, useEffect } from 'react';
import StrategyModal from '../components/StrategyModal';
import PairAnalysisDrawer from '../components/PairAnalysisDrawer';
import BacktestingDashboard from '../components/BacktestingDashboard';
import WorkflowDashboard from '../components/WorkflowDashboard';
import TickDataVisualization from '../components/TickDataVisualization';
import { apiService, Strategy, MarketData } from '../lib/api';

export default function TradingDashboard() {
  const [activeTab, setActiveTab] = useState<'overview' | 'market' | 'strategies' | 'portfolio' | 'analytics' | 'backtesting' | 'workflow' | 'tickdata' | 'settings'>('overview');
  const [selectedPair, setSelectedPair] = useState<string | null>(null);
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [showPairAnalysis, setShowPairAnalysis] = useState(false);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [performance, setPerformance] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState<string>('');
  const [economicEvents, setEconomicEvents] = useState<any[]>([]);

  const goTo = (tab: typeof activeTab) => setActiveTab(tab);

  // Update current time
  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(new Date().toLocaleTimeString());
    };
    
    updateTime();
    const interval = setInterval(updateTime, 1000);
    
    return () => clearInterval(interval);
  }, []);

  // Load initial data
  useEffect(() => {
    const loadData = async () => {
           setLoading(true);
           try {
             const [strategiesData, marketDataResponse, performanceData, eventsData] = await Promise.all([
               apiService.getStrategies().catch(() => []),
               apiService.getMarketData().catch(() => {
                 // Fallback mock data if API fails
                 return [
                   {
                     symbol: "EUR/USD",
                     price: 1.08542,
                     currentPrice: 1.08542,
                     change24h: 0.00123,
                     changePercent24h: 0.11,
                     volume: 1234567,
                     high24h: 1.08750,
                     low24h: 1.08320,
                     timestamp: new Date().toISOString()
                   },
                   {
                     symbol: "GBP/USD",
                     price: 1.26478,
                     currentPrice: 1.26478,
                     change24h: -0.00234,
                     changePercent24h: -0.18,
                     volume: 987654,
                     high24h: 1.26890,
                     low24h: 1.26210,
                     timestamp: new Date().toISOString()
                   },
                   {
                     symbol: "USD/JPY",
                     price: 149.123,
                     currentPrice: 149.123,
                     change24h: 0.456,
                     changePercent24h: 0.31,
                     volume: 2345678,
                     high24h: 149.890,
                     low24h: 148.750,
                     timestamp: new Date().toISOString()
                   },
                   {
                     symbol: "USD/CHF",
                     price: 0.8750,
                     currentPrice: 0.8750,
                     change24h: -0.0010,
                     changePercent24h: -0.11,
                     volume: 750000,
                     high24h: 0.8765,
                     low24h: 0.8735,
                     timestamp: new Date().toISOString()
                   },
                   {
                     symbol: "AUD/USD",
                     price: 0.6520,
                     currentPrice: 0.6520,
                     change24h: 0.0015,
                     changePercent24h: 0.23,
                     volume: 650000,
                     high24h: 0.6545,
                     low24h: 0.6495,
                     timestamp: new Date().toISOString()
                   },
                   {
                     symbol: "USD/CAD",
                     price: 1.3650,
                     currentPrice: 1.3650,
                     change24h: -0.0020,
                     changePercent24h: -0.15,
                     volume: 580000,
                     high24h: 1.3680,
                     low24h: 1.3620,
                     timestamp: new Date().toISOString()
                   }
                 ];
               }),
               apiService.getPerformanceSummary().catch(() => ({
                 totalValue: 12450.00,
                 totalReturn: 245.30,
                 totalReturnPercent: 2.01,
                 activeStrategies: 3,
                 totalTrades: 47,
                 winRate: 68.1
               })),
               fetch('http://localhost:8000/api/v1/economic-calendar/events/simple').then(res => res.json()).catch(() => ({ events: [] }))
             ]);
             
                         // Transform strategies data to extract the array
            const strategiesArray = Array.isArray(strategiesData) ? strategiesData : [];
            setStrategies(strategiesArray);
            setMarketData(marketDataResponse);
            setEconomicEvents(eventsData.events || []);
            
            // Transform performance data to match expected format
            let transformedPerformance = performanceData;
            if (performanceData && typeof performanceData === 'object' && 'data' in performanceData) {
              // Backend returns different format, transform it
              const data = (performanceData as any).data;
              transformedPerformance = {
                totalValue: data.top_performer?.current_performance || 10000,
                totalReturn: data.top_performer?.current_performance - 10000 || 0,
                totalReturnPercent: ((data.top_performer?.current_performance - 10000) / 10000 * 100) || 0,
                activeStrategies: data.total_strategies || 0,
                totalTrades: data.top_performer?.total_trades || 0,
                winRate: data.top_performer?.win_rate || 0
              };
            }
             
             setPerformance(transformedPerformance);
           } catch (err) {
             console.error('Failed to load dashboard data:', err);
             setError('Failed to load dashboard data');
             // Set fallback performance data
             setPerformance({
               totalValue: 10000,
               totalReturn: 0,
               totalReturnPercent: 0,
               activeStrategies: 0,
               totalTrades: 0,
               winRate: 0
             });
           } finally {
             setLoading(false);
           }
         };

    loadData();
  }, []);

  const handleGenerateStrategy = (strategy: Strategy) => {
    setStrategies(prev => [...prev, strategy]);
  };

  const handleAnalyzePair = (pair: string) => {
    setSelectedPair(pair);
    setShowPairAnalysis(true);
  };

  return (
    <>
      <Head>
        <title>AXF Bot 0 - AI-Powered Forex Trading Dashboard</title>
        <meta name="description" content="Professional AI-powered forex trading system with real-time market data, strategy generation, and portfolio management" />
      </Head>

      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center py-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold text-sm">AXF</span>
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">AXF Bot 0</h1>
                    <p className="text-sm text-gray-600">AI-Powered Forex Trading</p>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full bg-green-500"></div>
                  <span className="text-sm text-gray-600">System Online</span>
                </div>
                
                <div className="text-right">
                  <p className="text-sm text-gray-500">Last Update</p>
                  <p className="text-sm font-medium text-gray-900">
                    {currentTime || '...'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <nav className="flex space-x-8">
              <button onClick={() => goTo('overview')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'overview' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">📊</span>
                Overview
              </button>
              <button onClick={() => goTo('market')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'market' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">💹</span>
                Market Data
              </button>
              <button onClick={() => goTo('strategies')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'strategies' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">🤖</span>
                AI Strategies
              </button>
              <button onClick={() => goTo('portfolio')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'portfolio' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">💼</span>
                Portfolio
              </button>
              <button onClick={() => goTo('analytics')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'analytics' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">📈</span>
                Analytics
              </button>
              <button onClick={() => goTo('backtesting')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'backtesting' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">🧪</span>
                Backtesting
              </button>
              <button onClick={() => goTo('workflow')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'workflow' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">🔄</span>
                Workflow
              </button>
              <button onClick={() => goTo('tickdata')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'tickdata' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">📊</span>
                Tick Data
              </button>
              <button onClick={() => goTo('settings')} className={`py-4 px-1 border-b-2 font-medium text-sm ${activeTab === 'settings' ? 'border-blue-500 text-blue-600' : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'}`}>
                <span className="mr-2">⚙️</span>
                Settings
              </button>
            </nav>
          </div>
        </div>

        {/* Main Content */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* OVERVIEW */}
          {activeTab === 'overview' && (
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
                                {loading ? '...' : marketData.length}
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
                              <dd className="text-lg font-medium text-gray-900">
                                                                 {loading ? '...' : (Array.isArray(strategies) ? strategies.filter(s => s.status === 'active').length : 0)}
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
                            <div className="w-8 h-8 bg-purple-500 rounded-md flex items-center justify-center">
                              <span className="text-white font-bold">💰</span>
                            </div>
                          </div>
                          <div className="ml-5 w-0 flex-1">
                            <dl>
                                                             <dt className="text-sm font-medium text-gray-500 truncate">Portfolio Value</dt>
                               <dd className="text-lg font-medium text-gray-900">
                                 {loading ? '...' : performance && typeof performance.totalValue === 'number' ? `$${performance.totalValue.toLocaleString()}` : '$0.00'}
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
                              <span className="text-white font-bold">📈</span>
                            </div>
                          </div>
                          <div className="ml-5 w-0 flex-1">
                            <dl>
                              <dt className="text-sm font-medium text-gray-500 truncate">24h P&L</dt>
                              <dd className={`text-lg font-medium ${
                                performance && typeof performance.totalReturn === 'number' && performance.totalReturn >= 0 ? 'text-green-600' : 'text-red-600'
                              }`}>
                                {loading ? '...' : performance && typeof performance.totalReturn === 'number' ? 
                                  `${performance.totalReturn >= 0 ? '+' : ''}$${performance.totalReturn.toFixed(2)}` : 
                                  '$0.00'
                                }
                              </dd>
                            </dl>
                          </div>
                        </div>
                      </div>
                    </div>
            </div>

            {/* Economic Events */}
            {economicEvents.length > 0 && (
              <div className="bg-white shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">📅 Upcoming Economic Events</h3>
                  <div className="space-y-3">
                    {economicEvents.slice(0, 3).map((event, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex-1">
                          <div className="flex items-center space-x-2">
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              event.impact === 'high' ? 'bg-red-100 text-red-800' :
                              event.impact === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-green-100 text-green-800'
                            }`}>
                              {event.impact?.toUpperCase() || 'LOW'}
                            </span>
                            <span className="text-sm font-medium text-gray-900">{event.title}</span>
                            <span className="text-xs text-gray-500">({event.country})</span>
                          </div>
                          <div className="text-xs text-gray-600 mt-1">
                            {event.event_time ? new Date(event.event_time).toLocaleString() : 'Time TBD'}
                          </div>
                          {event.trading_opportunities && event.trading_opportunities.length > 0 && (
                            <div className="text-xs text-blue-600 mt-1">
                              💡 {event.trading_opportunities[0]}
                            </div>
                          )}
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-medium text-gray-900">
                            {event.market_impact_score ? `${Math.round(event.market_impact_score * 100)}%` : 'N/A'}
                          </div>
                          <div className="text-xs text-gray-500">Impact</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Quick Actions */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Quick Actions</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <button onClick={() => setShowStrategyModal(true)} className="bg-blue-600 text-white px-4 py-3 rounded-md hover:bg-blue-700 transition-colors text-left">
                    <div className="flex items-center">
                      <span className="text-2xl mr-3">🤖</span>
                      <div>
                        <div className="font-medium">Generate Strategy</div>
                        <div className="text-sm opacity-90">AI-powered strategy creation</div>
                      </div>
                    </div>
                  </button>
                  
                  <button onClick={() => goTo('market')} className="bg-green-600 text-white px-4 py-3 rounded-md hover:bg-green-700 transition-colors text-left">
                    <div className="flex items-center">
                      <span className="text-2xl mr-3">📊</span>
                      <div>
                        <div className="font-medium">Market Analysis</div>
                        <div className="text-sm opacity-90">Real-time market insights</div>
                      </div>
                    </div>
                  </button>
                  
                  <button onClick={() => goTo('settings')} className="bg-purple-600 text-white px-4 py-3 rounded-md hover:bg-purple-700 transition-colors text-left">
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
          )}

          {/* MARKET DATA */}
          {activeTab === 'market' && (
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
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {loading ? (
                        <tr>
                          <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                            <div className="flex items-center justify-center">
                              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mr-3"></div>
                              Loading market data...
                            </div>
                          </td>
                        </tr>
                      ) : marketData.length > 0 ? (
                        marketData.map((pair) => (
                          <tr key={pair.symbol} className="hover:bg-gray-50">
                            <td className="px-4 py-4 whitespace-nowrap">
                              <div className="text-sm font-medium text-gray-900">{pair.symbol}</div>
                            </td>
                            <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">
                              {typeof pair.price === 'number' ? pair.price.toFixed(5) : '0.00000'}
                            </td>
                            <td className={`px-4 py-4 whitespace-nowrap text-sm ${
                              typeof pair.change24h === 'number' && pair.change24h >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {typeof pair.change24h === 'number' ? (pair.change24h >= 0 ? '+' : '') + pair.change24h.toFixed(5) : '+0.00000'}
                            </td>
                            <td className={`px-4 py-4 whitespace-nowrap text-sm ${
                              typeof pair.changePercent24h === 'number' && pair.changePercent24h >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              {typeof pair.changePercent24h === 'number' ? (pair.changePercent24h >= 0 ? '+' : '') + pair.changePercent24h.toFixed(2) + '%' : '+0.00%'}
                            </td>
                            <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-900">
                              {typeof pair.volume === 'number' ? pair.volume.toLocaleString() : '0'}
                            </td>
                            <td className="px-4 py-4 whitespace-nowrap text-sm font-medium">
                              <button 
                                onClick={() => handleAnalyzePair(pair.symbol)} 
                                className="text-blue-600 hover:text-blue-900"
                              >
                                Analyze
                              </button>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                            {error ? `Error: ${error}` : 'No market data available'}
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                  {selectedPair && (
                    <div className="mt-4 text-sm text-gray-700">
                      Selected pair: <span className="font-medium">{selectedPair}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* AI STRATEGIES */}
          {activeTab === 'strategies' && (
            <div className="space-y-6">
              <div className="bg-white shadow rounded-lg">
                <div className="px-4 py-5 sm:p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg leading-6 font-medium text-gray-900">AI Strategy Generation</h3>
                    <button
                      onClick={() => setShowStrategyModal(true)}
                      className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
                    >
                      + Generate New Strategy
                    </button>
                  </div>
                  
                  {Array.isArray(strategies) && strategies.length > 0 ? (
                    <div className="space-y-4">
                      <h4 className="text-md font-semibold text-gray-900">AI Strategies</h4>
                      {strategies.map((strategy) => (
                        <div key={strategy.id} className="border rounded-lg p-4">
                          <div className="flex justify-between items-start">
                            <div className="flex-1">
                              <h5 className="font-medium text-gray-900">{strategy.name}</h5>
                              <p className="text-sm text-gray-600 mb-2">{strategy.description}</p>
                              <div className="flex flex-wrap gap-2 text-xs text-gray-500">
                                <span className="px-2 py-1 bg-gray-100 rounded">
                                  {strategy.parameters?.marketConditions || 'Unknown'}
                                </span>
                                <span className="px-2 py-1 bg-gray-100 rounded">
                                  {strategy.parameters?.riskLevel || 'Unknown'}
                                </span>
                                <span className="px-2 py-1 bg-gray-100 rounded">
                                  {strategy.parameters?.timeframe || 'Unknown'}
                                </span>
                              </div>
                              <p className="text-xs text-gray-500 mt-2">
                                Created: {strategy.createdAt ? new Date(strategy.createdAt).toLocaleString() : 'Unknown'}
                              </p>
                              {strategy.performance && (
                                <div className="mt-2 flex gap-4 text-xs">
                                  <span>Return: {typeof strategy.performance.totalReturn === 'number' ? strategy.performance.totalReturn.toFixed(2) : '0.00'}%</span>
                                  <span>Sharpe: {typeof strategy.performance.sharpeRatio === 'number' ? strategy.performance.sharpeRatio.toFixed(2) : '0.00'}</span>
                                  <span>Max DD: {typeof strategy.performance.maxDrawdown === 'number' ? strategy.performance.maxDrawdown.toFixed(2) : '0.00'}%</span>
                                </div>
                              )}
                            </div>
                            <span className={`px-2 py-1 text-xs rounded-full ${
                              strategy.status === 'active' ? 'bg-green-100 text-green-800' :
                              strategy.status === 'testing' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-gray-100 text-gray-800'
                            }`}>
                              {strategy.status}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="text-gray-400 text-4xl mb-4">🤖</div>
                      <h4 className="text-lg font-medium text-gray-900 mb-2">No Strategies Generated Yet</h4>
                      <p className="text-gray-600 mb-4">
                        Click "Generate New Strategy" to create your first AI-powered trading strategy.
                      </p>
                      <button
                        onClick={() => setShowStrategyModal(true)}
                        className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 transition-colors"
                      >
                        Generate Your First Strategy
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* PORTFOLIO */}
          {activeTab === 'portfolio' && (
            <div className="bg-white shadow rounded-lg p-6 text-gray-700">Portfolio view coming soon.</div>
          )}
          {/* ANALYTICS */}
          {activeTab === 'analytics' && (
            <div className="bg-white shadow rounded-lg p-6 text-gray-700">Analytics view coming soon.</div>
          )}
          {/* BACKTESTING */}
          {activeTab === 'backtesting' && (
            <BacktestingDashboard />
          )}
          {/* WORKFLOW */}
          {activeTab === 'workflow' && (
            <WorkflowDashboard />
          )}
          {/* TICK DATA */}
          {activeTab === 'tickdata' && (
            <TickDataVisualization symbol="EURUSD" days={1} />
          )}
          {/* SETTINGS */}
          {activeTab === 'settings' && (
            <div className="bg-white shadow rounded-lg p-6 text-gray-700">Settings view coming soon.</div>
          )}
        </div>
      </div>

      {/* Modals and Overlays */}
      <StrategyModal
        isOpen={showStrategyModal}
        onClose={() => setShowStrategyModal(false)}
        onGenerate={handleGenerateStrategy}
      />
      
      <PairAnalysisDrawer
        isOpen={showPairAnalysis}
        onClose={() => setShowPairAnalysis(false)}
        pair={selectedPair}
      />
    </>
  );
}