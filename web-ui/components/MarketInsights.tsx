import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { 
  TrendingUpIcon, 
  TrendingDownIcon, 
  ArrowUpIcon, 
  ArrowDownIcon,
  CurrencyDollarIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

interface MarketData {
  currencyPairs: Array<{
    symbol: string;
    currentPrice: number;
    change24h: number;
    changePercent24h: number;
    volume: number;
  }>;
  sentiment: {
    overall: number;
    news: number;
    social: number;
    technical: number;
  };
  economicEvents: Array<{
    name: string;
    time: string;
    impact: 'low' | 'medium' | 'high';
    currency: string;
  }>;
  priceHistory: Array<{
    time: string;
    eurusd: number;
    gbpusd: number;
    usdjpy: number;
  }>;
}

interface MarketInsightsProps {
  data?: MarketData;
  detailed?: boolean;
}

export default function MarketInsights({ data, detailed = false }: MarketInsightsProps) {
  const [selectedPair, setSelectedPair] = useState('EURUSD');

  if (!data) {
    return (
      <div className="card">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-3 bg-gray-200 rounded"></div>
            <div className="h-3 bg-gray-200 rounded w-5/6"></div>
            <div className="h-3 bg-gray-200 rounded w-4/6"></div>
          </div>
        </div>
      </div>
    );
  }

  const formatPrice = (price: number) => price.toFixed(5);
  const formatPercent = (percent: number) => `${percent >= 0 ? '+' : ''}${percent.toFixed(2)}%`;

  const getChangeColor = (change: number) => {
    return change >= 0 ? 'text-success-600' : 'text-danger-600';
  };

  const getChangeIcon = (change: number) => {
    return change >= 0 ? (
      <ArrowUpIcon className="h-4 w-4" />
    ) : (
      <ArrowDownIcon className="h-4 w-4" />
    );
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'bg-danger-100 text-danger-800';
      case 'medium': return 'bg-warning-100 text-warning-800';
      case 'low': return 'bg-success-100 text-success-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Currency Pairs */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Major Currency Pairs</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.currencyPairs.slice(0, detailed ? 9 : 6).map((pair) => (
            <div
              key={pair.symbol}
              className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                selectedPair === pair.symbol
                  ? 'border-primary-500 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setSelectedPair(pair.symbol)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{pair.symbol}</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {formatPrice(pair.currentPrice)}
                  </p>
                </div>
                <div className={`flex items-center space-x-1 ${getChangeColor(pair.changePercent24h)}`}>
                  {getChangeIcon(pair.changePercent24h)}
                  <span className="text-sm font-medium">
                    {formatPercent(pair.changePercent24h)}
                  </span>
                </div>
              </div>
              <div className="mt-2 text-sm text-gray-500">
                Vol: {pair.volume.toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Price Chart */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          {selectedPair} Price Chart (24h)
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.priceHistory}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="time" 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => formatPrice(value)}
              />
              <Tooltip
                formatter={(value: number) => [formatPrice(value), selectedPair]}
                labelFormatter={(value) => new Date(value).toLocaleString()}
              />
              <Area
                type="monotone"
                dataKey={selectedPair.toLowerCase()}
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.1}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Market Sentiment */}
      <div className="card">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Market Sentiment</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {data.sentiment.overall.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">Overall</div>
            <div className={`w-full bg-gray-200 rounded-full h-2 mt-2 ${
              data.sentiment.overall >= 0 ? 'bg-success-200' : 'bg-danger-200'
            }`}>
              <div
                className={`h-2 rounded-full ${
                  data.sentiment.overall >= 0 ? 'bg-success-500' : 'bg-danger-500'
                }`}
                style={{ width: `${Math.abs(data.sentiment.overall)}%` }}
              />
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {data.sentiment.news.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">News</div>
            <div className={`w-full bg-gray-200 rounded-full h-2 mt-2 ${
              data.sentiment.news >= 0 ? 'bg-success-200' : 'bg-danger-200'
            }`}>
              <div
                className={`h-2 rounded-full ${
                  data.sentiment.news >= 0 ? 'bg-success-500' : 'bg-danger-500'
                }`}
                style={{ width: `${Math.abs(data.sentiment.news)}%` }}
              />
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {data.sentiment.social.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">Social</div>
            <div className={`w-full bg-gray-200 rounded-full h-2 mt-2 ${
              data.sentiment.social >= 0 ? 'bg-success-200' : 'bg-danger-200'
            }`}>
              <div
                className={`h-2 rounded-full ${
                  data.sentiment.social >= 0 ? 'bg-success-500' : 'bg-danger-500'
                }`}
                style={{ width: `${Math.abs(data.sentiment.social)}%` }}
              />
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {data.sentiment.technical.toFixed(1)}
            </div>
            <div className="text-sm text-gray-500">Technical</div>
            <div className={`w-full bg-gray-200 rounded-full h-2 mt-2 ${
              data.sentiment.technical >= 0 ? 'bg-success-200' : 'bg-danger-200'
            }`}>
              <div
                className={`h-2 rounded-full ${
                  data.sentiment.technical >= 0 ? 'bg-success-500' : 'bg-danger-500'
                }`}
                style={{ width: `${Math.abs(data.sentiment.technical)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Economic Events */}
      {detailed && (
        <div className="card">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Upcoming Economic Events</h3>
          <div className="space-y-3">
            {data.economicEvents.map((event, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-900">{event.name}</span>
                    <span className={`badge ${getImpactColor(event.impact)}`}>
                      {event.impact.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm text-gray-500">
                    {event.currency} • {event.time}
                  </div>
                </div>
                <ClockIcon className="h-5 w-5 text-gray-400" />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
