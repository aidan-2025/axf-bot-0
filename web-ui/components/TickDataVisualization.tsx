import React, { useState, useEffect } from 'react';

interface TickData {
  timestamp: string;
  bid: number;
  ask: number;
  spread: number;
  mid: number;
  volume: number;
}

interface TickDataVisualizationProps {
  symbol?: string;
  days?: number;
}

const TickDataVisualization: React.FC<TickDataVisualizationProps> = ({ 
  symbol = "EURUSD", 
  days = 1 
}) => {
  const [tickData, setTickData] = useState<TickData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1m' | '5m' | '15m' | '1h'>('1m');
  const [showSpread, setShowSpread] = useState(true);
  const [showVolume, setShowVolume] = useState(false);

  // Generate sample tick data (simulating real tick data)
  const generateSampleTickData = (symbol: string, days: number): TickData[] => {
    const data: TickData[] = [];
    const startTime = new Date();
    startTime.setDate(startTime.getDate() - days);
    
    const ticksPerDay = 86400; // 1 tick per second
    const totalTicks = days * ticksPerDay;
    
    let basePrice = 1.1000;
    if (symbol.includes('GBP')) basePrice = 1.2600;
    if (symbol.includes('JPY')) basePrice = 149.000;
    if (symbol.includes('CHF')) basePrice = 0.8750;
    if (symbol.includes('AUD')) basePrice = 0.6520;
    if (symbol.includes('CAD')) basePrice = 1.3650;
    
    for (let i = 0; i < totalTicks; i++) {
      const timestamp = new Date(startTime.getTime() + i * 1000);
      
      // Generate realistic price movement
      const priceChange = (Math.random() - 0.5) * 0.0001;
      basePrice += priceChange;
      
      // Generate realistic spread (1-3 pips)
      const spread = 0.0001 + Math.random() * 0.0002;
      
      const bid = basePrice - spread / 2;
      const ask = basePrice + spread / 2;
      const mid = (bid + ask) / 2;
      
      // Generate volume (higher during market hours)
      const hour = timestamp.getHours();
      const isMarketHours = (hour >= 8 && hour <= 17) || (hour >= 21 && hour <= 23);
      const volume = isMarketHours ? Math.floor(Math.random() * 100) + 10 : Math.floor(Math.random() * 20) + 1;
      
      data.push({
        timestamp: timestamp.toISOString(),
        bid: parseFloat(bid.toFixed(5)),
        ask: parseFloat(ask.toFixed(5)),
        spread: parseFloat(spread.toFixed(5)),
        mid: parseFloat(mid.toFixed(5)),
        volume
      });
    }
    
    return data;
  };

  useEffect(() => {
    const loadTickData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Generate sample data
        const data = generateSampleTickData(symbol, days);
        setTickData(data);
      } catch (err) {
        setError('Failed to load tick data');
        console.error('Error loading tick data:', err);
      } finally {
        setLoading(false);
      }
    };

    loadTickData();
  }, [symbol, days]);

  // Aggregate data based on selected timeframe
  const aggregateData = (data: TickData[], timeframe: string) => {
    const interval = timeframe === '1m' ? 60 : 
                    timeframe === '5m' ? 300 : 
                    timeframe === '15m' ? 900 : 3600;
    
    const aggregated: { [key: string]: TickData[] } = {};
    
    data.forEach(tick => {
      const time = new Date(tick.timestamp);
      const key = Math.floor(time.getTime() / (interval * 1000)) * (interval * 1000);
      const keyStr = new Date(key).toISOString();
      
      if (!aggregated[keyStr]) {
        aggregated[keyStr] = [];
      }
      aggregated[keyStr].push(tick);
    });
    
    return Object.entries(aggregated).map(([timestamp, ticks]) => {
      const bid = ticks.reduce((sum, t) => sum + t.bid, 0) / ticks.length;
      const ask = ticks.reduce((sum, t) => sum + t.ask, 0) / ticks.length;
      const mid = ticks.reduce((sum, t) => sum + t.mid, 0) / ticks.length;
      const spread = ticks.reduce((sum, t) => sum + t.spread, 0) / ticks.length;
      const volume = ticks.reduce((sum, t) => sum + t.volume, 0);
      
      return {
        timestamp,
        bid: parseFloat(bid.toFixed(5)),
        ask: parseFloat(ask.toFixed(5)),
        spread: parseFloat(spread.toFixed(5)),
        mid: parseFloat(mid.toFixed(5)),
        volume
      };
    }).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  };

  const aggregatedData = aggregateData(tickData, selectedTimeframe);
  const latestTick = tickData[tickData.length - 1];
  const avgSpread = tickData.reduce((sum, t) => sum + t.spread, 0) / tickData.length;
  const totalVolume = tickData.reduce((sum, t) => sum + t.volume, 0);

  if (loading) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mr-3"></div>
          <span className="text-gray-600">Loading tick data...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white shadow rounded-lg p-6">
        <div className="text-center text-red-600">
          <div className="text-4xl mb-2">⚠️</div>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white shadow rounded-lg">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Tick Data Visualization</h3>
            <p className="text-sm text-gray-600">{symbol} - {days} day{days > 1 ? 's' : ''} of data</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-600">Timeframe:</label>
              <select
                value={selectedTimeframe}
                onChange={(e) => setSelectedTimeframe(e.target.value as any)}
                className="border border-gray-300 rounded-md px-3 py-1 text-sm"
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="1h">1 Hour</option>
              </select>
            </div>
            <div className="flex items-center space-x-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showSpread}
                  onChange={(e) => setShowSpread(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">Show Spread</span>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showVolume}
                  onChange={(e) => setShowVolume(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">Show Volume</span>
              </label>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="text-sm text-blue-600 font-medium">Current Price</div>
            <div className="text-2xl font-bold text-blue-900">
              {latestTick ? latestTick.mid.toFixed(5) : '0.00000'}
            </div>
            <div className="text-xs text-blue-600">
              Bid: {latestTick ? latestTick.bid.toFixed(5) : '0.00000'} | 
              Ask: {latestTick ? latestTick.ask.toFixed(5) : '0.00000'}
            </div>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="text-sm text-green-600 font-medium">Average Spread</div>
            <div className="text-2xl font-bold text-green-900">
              {(avgSpread * 10000).toFixed(1)} pips
            </div>
            <div className="text-xs text-green-600">
              {avgSpread.toFixed(5)}
            </div>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="text-sm text-purple-600 font-medium">Total Ticks</div>
            <div className="text-2xl font-bold text-purple-900">
              {tickData.length.toLocaleString()}
            </div>
            <div className="text-xs text-purple-600">
              {aggregatedData.length} aggregated bars
            </div>
          </div>
          
          <div className="bg-orange-50 p-4 rounded-lg">
            <div className="text-sm text-orange-600 font-medium">Total Volume</div>
            <div className="text-2xl font-bold text-orange-900">
              {totalVolume.toLocaleString()}
            </div>
            <div className="text-xs text-orange-600">
              Avg: {Math.round(totalVolume / tickData.length)}
            </div>
          </div>
        </div>

        {/* Price Chart */}
        <div className="mb-6">
          <h4 className="text-md font-medium text-gray-900 mb-3">Price Movement</h4>
          <div className="h-64 bg-gray-50 rounded-lg p-4 relative">
            <div className="h-full flex items-end space-x-1">
              {aggregatedData.slice(-100).map((tick, index) => {
                const height = ((tick.mid - Math.min(...aggregatedData.slice(-100).map(t => t.mid))) / 
                              (Math.max(...aggregatedData.slice(-100).map(t => t.mid)) - 
                               Math.min(...aggregatedData.slice(-100).map(t => t.mid)))) * 100;
                return (
                  <div
                    key={index}
                    className="bg-blue-500 rounded-sm flex-1"
                    style={{ height: `${Math.max(height, 1)}%` }}
                    title={`${new Date(tick.timestamp).toLocaleTimeString()}: ${tick.mid.toFixed(5)}`}
                  />
                );
              })}
            </div>
          </div>
        </div>

        {/* Spread Chart */}
        {showSpread && (
          <div className="mb-6">
            <h4 className="text-md font-medium text-gray-900 mb-3">Spread Distribution</h4>
            <div className="h-32 bg-gray-50 rounded-lg p-4 relative">
              <div className="h-full flex items-end space-x-1">
                {aggregatedData.slice(-100).map((tick, index) => {
                  const height = (tick.spread / Math.max(...aggregatedData.slice(-100).map(t => t.spread))) * 100;
                  return (
                    <div
                      key={index}
                      className="bg-red-500 rounded-sm flex-1"
                      style={{ height: `${Math.max(height, 1)}%` }}
                      title={`${new Date(tick.timestamp).toLocaleTimeString()}: ${(tick.spread * 10000).toFixed(1)} pips`}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Volume Chart */}
        {showVolume && (
          <div className="mb-6">
            <h4 className="text-md font-medium text-gray-900 mb-3">Volume Profile</h4>
            <div className="h-32 bg-gray-50 rounded-lg p-4 relative">
              <div className="h-full flex items-end space-x-1">
                {aggregatedData.slice(-100).map((tick, index) => {
                  const height = (tick.volume / Math.max(...aggregatedData.slice(-100).map(t => t.volume))) * 100;
                  return (
                    <div
                      key={index}
                      className="bg-green-500 rounded-sm flex-1"
                      style={{ height: `${Math.max(height, 1)}%` }}
                      title={`${new Date(tick.timestamp).toLocaleTimeString()}: ${tick.volume}`}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* Recent Ticks Table */}
        <div>
          <h4 className="text-md font-medium text-gray-900 mb-3">Recent Ticks</h4>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Bid</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Ask</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Mid</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Spread (pips)</th>
                  <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase">Volume</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {aggregatedData.slice(-10).reverse().map((tick, index) => (
                  <tr key={index} className="hover:bg-gray-50">
                    <td className="px-4 py-2 text-sm text-gray-900">
                      {new Date(tick.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-4 py-2 text-sm text-gray-900">{tick.bid.toFixed(5)}</td>
                    <td className="px-4 py-2 text-sm text-gray-900">{tick.ask.toFixed(5)}</td>
                    <td className="px-4 py-2 text-sm text-gray-900">{tick.mid.toFixed(5)}</td>
                    <td className="px-4 py-2 text-sm text-gray-900">{(tick.spread * 10000).toFixed(1)}</td>
                    <td className="px-4 py-2 text-sm text-gray-900">{tick.volume}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TickDataVisualization;

