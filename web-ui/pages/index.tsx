import { useState, useEffect } from 'react';
import Head from 'next/head';
import { useQuery } from 'react-query';
import { 
  ChartBarIcon, 
  CurrencyDollarIcon, 
  TrendingUpIcon, 
  ClockIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline';
import Layout from '../components/Layout';
import MarketInsights from '../components/MarketInsights';
import StrategyList from '../components/StrategyList';
import PerformanceMetrics from '../components/PerformanceMetrics';
import RecentStrategies from '../components/RecentStrategies';
import { fetchSystemHealth, fetchMarketData, fetchStrategies, fetchPerformance } from '../lib/api';

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  // Fetch system health
  const { data: health, isLoading: healthLoading } = useQuery(
    'systemHealth',
    fetchSystemHealth,
    { refetchInterval: 30000 }
  );

  // Fetch market data
  const { data: marketData, isLoading: marketLoading } = useQuery(
    'marketData',
    fetchMarketData,
    { refetchInterval: 60000 }
  );

  // Fetch strategies
  const { data: strategies, isLoading: strategiesLoading } = useQuery(
    'strategies',
    fetchStrategies,
    { refetchInterval: 120000 }
  );

  // Fetch performance data
  const { data: performance, isLoading: performanceLoading } = useQuery(
    'performance',
    fetchPerformance,
    { refetchInterval: 300000 }
  );

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'strategies', name: 'Strategies', icon: TrendingUpIcon },
    { id: 'performance', name: 'Performance', icon: CurrencyDollarIcon },
    { id: 'market', name: 'Market Insights', icon: ClockIcon },
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <TrendingUpIcon className="h-8 w-8 text-success-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-500">Active Strategies</p>
                    <p className="text-2xl font-semibold text-gray-900">
                      {strategies?.active || 0}
                    </p>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <CurrencyDollarIcon className="h-8 w-8 text-primary-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-500">Total Profit</p>
                    <p className="text-2xl font-semibold text-gray-900">
                      ${performance?.totalProfit?.toFixed(2) || '0.00'}
                    </p>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <ChartBarIcon className="h-8 w-8 text-warning-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-500">Win Rate</p>
                    <p className="text-2xl font-semibold text-gray-900">
                      {performance?.winRate?.toFixed(1) || '0.0'}%
                    </p>
                  </div>
                </div>
              </div>

              <div className="card">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <ExclamationTriangleIcon className="h-8 w-8 text-danger-600" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-500">Max Drawdown</p>
                    <p className="text-2xl font-semibold text-gray-900">
                      {performance?.maxDrawdown?.toFixed(1) || '0.0'}%
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <RecentStrategies strategies={strategies?.recent || []} />
              <MarketInsights data={marketData} />
            </div>
          </div>
        );

      case 'strategies':
        return <StrategyList strategies={strategies?.all || []} />;

      case 'performance':
        return <PerformanceMetrics data={performance} />;

      case 'market':
        return <MarketInsights data={marketData} detailed />;

      default:
        return null;
    }
  };

  return (
    <>
      <Head>
        <title>AXF Bot 0 - AI-Powered Forex Trading System</title>
        <meta name="description" content="Advanced AI-powered forex trading system with real-time strategy generation and performance monitoring" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Layout>
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
                      health?.status === 'healthy' ? 'bg-success-500' : 'bg-danger-500'
                    }`} />
                    <span className="text-sm text-gray-600">
                      {health?.status === 'healthy' ? 'System Healthy' : 'System Issues'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Navigation Tabs */}
              <div className="border-b border-gray-200">
                <nav className="-mb-px flex space-x-8">
                  {tabs.map((tab) => {
                    const Icon = tab.icon;
                    return (
                      <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 ${
                          activeTab === tab.id
                            ? 'border-primary-500 text-primary-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                        }`}
                      >
                        <Icon className="h-5 w-5" />
                        <span>{tab.name}</span>
                      </button>
                    );
                  })}
                </nav>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            {renderTabContent()}
          </div>
        </div>
      </Layout>
    </>
  );
}
