import { useState, useEffect } from 'react';
import { apiService, PairAnalysis } from '../lib/api';

interface PairAnalysisDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  pair: string | null;
}

export default function PairAnalysisDrawer({ isOpen, onClose, pair }: PairAnalysisDrawerProps) {
  const [pairData, setPairData] = useState<PairAnalysis | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && pair) {
      setLoading(true);
      setError(null);
      
      const fetchAnalysis = async () => {
        try {
          const analysis = await apiService.analyzePair(pair);
          setPairData(analysis);
        } catch (err) {
          console.error('Failed to fetch pair analysis:', err);
          setError('Failed to load analysis data');
          
          // Fallback to mock data
          const mockData: PairAnalysis = {
            symbol: pair,
            price: pair === 'EUR/USD' ? 1.08542 : pair === 'GBP/USD' ? 1.26478 : 149.123,
            change24h: pair === 'EUR/USD' ? 0.00123 : pair === 'GBP/USD' ? -0.00234 : 0.456,
            changePercent24h: pair === 'EUR/USD' ? 0.11 : pair === 'GBP/USD' ? -0.18 : 0.31,
            volume: pair === 'EUR/USD' ? 1234567 : pair === 'GBP/USD' ? 987654 : 2345678,
            high24h: pair === 'EUR/USD' ? 1.08750 : pair === 'GBP/USD' ? 1.26890 : 149.890,
            low24h: pair === 'EUR/USD' ? 1.08320 : pair === 'GBP/USD' ? 1.26210 : 148.750,
            rsi: 65.2,
            macd: 0.0012,
            bollingerUpper: pair === 'EUR/USD' ? 1.08900 : pair === 'GBP/USD' ? 1.27100 : 150.200,
            bollingerLower: pair === 'EUR/USD' ? 1.08200 : pair === 'GBP/USD' ? 1.25800 : 148.100,
            support: pair === 'EUR/USD' ? 1.08300 : pair === 'GBP/USD' ? 1.26000 : 148.500,
            resistance: pair === 'EUR/USD' ? 1.08800 : pair === 'GBP/USD' ? 1.26800 : 149.500,
            trend: 'bullish',
            sentiment: 'positive',
            recommendations: [
              'Strong upward momentum detected. Consider long positions with proper risk management.',
              'RSI indicates overbought conditions. Monitor for potential reversal signals.',
              'MACD shows bullish divergence. Trend continuation likely.'
            ]
          };
          setPairData(mockData);
        } finally {
          setLoading(false);
        }
      };

      fetchAnalysis();
    }
  }, [isOpen, pair]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-end z-50">
      <div className="bg-white w-full max-w-md h-full overflow-y-auto">
        <div className="p-6 border-b border-gray-200">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold text-gray-900">
              {pair} Analysis
            </h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              ✕
            </button>
          </div>
        </div>

        <div className="p-6">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="ml-3 text-gray-600">Analyzing {pair}...</span>
            </div>
          ) : pairData ? (
            <div className="space-y-6">
              {/* Price Overview */}
              <div className="bg-gray-50 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-2xl font-bold text-gray-900">
                    {pairData.price.toFixed(5)}
                  </span>
                  <span className={`text-sm font-medium ${
                    pairData.change24h >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {pairData.change24h >= 0 ? '+' : ''}{pairData.change24h.toFixed(5)} 
                    ({pairData.changePercent24h >= 0 ? '+' : ''}{pairData.changePercent24h.toFixed(2)}%)
                  </span>
                </div>
                <div className="text-sm text-gray-600">
                  High: {pairData.high24h.toFixed(5)} | Low: {pairData.low24h.toFixed(5)}
                </div>
              </div>

              {/* Technical Indicators */}
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">Technical Indicators</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white border rounded-lg p-3">
                    <div className="text-sm text-gray-600">RSI (14)</div>
                    <div className="text-lg font-semibold text-gray-900">{pairData.rsi}</div>
                    <div className={`text-xs ${
                      pairData.rsi > 70 ? 'text-red-600' : pairData.rsi < 30 ? 'text-green-600' : 'text-gray-600'
                    }`}>
                      {pairData.rsi > 70 ? 'Overbought' : pairData.rsi < 30 ? 'Oversold' : 'Neutral'}
                    </div>
                  </div>
                  <div className="bg-white border rounded-lg p-3">
                    <div className="text-sm text-gray-600">MACD</div>
                    <div className="text-lg font-semibold text-gray-900">{pairData.macd.toFixed(4)}</div>
                    <div className={`text-xs ${
                      pairData.macd > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {pairData.macd > 0 ? 'Bullish' : 'Bearish'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Bollinger Bands */}
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">Bollinger Bands</h4>
                <div className="bg-white border rounded-lg p-3">
                  <div className="flex justify-between text-sm">
                    <div>
                      <span className="text-gray-600">Upper:</span>
                      <span className="ml-2 font-medium">{pairData.bollingerUpper.toFixed(5)}</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Lower:</span>
                      <span className="ml-2 font-medium">{pairData.bollingerLower.toFixed(5)}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Support & Resistance */}
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">Support & Resistance</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-white border rounded-lg p-3">
                    <div className="text-sm text-gray-600">Support</div>
                    <div className="text-lg font-semibold text-green-600">{pairData.support.toFixed(5)}</div>
                  </div>
                  <div className="bg-white border rounded-lg p-3">
                    <div className="text-sm text-gray-600">Resistance</div>
                    <div className="text-lg font-semibold text-red-600">{pairData.resistance.toFixed(5)}</div>
                  </div>
                </div>
              </div>

              {/* Market Sentiment */}
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">Market Analysis</h4>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Trend</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      pairData.trend === 'bullish' ? 'bg-green-100 text-green-800' :
                      pairData.trend === 'bearish' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {pairData.trend.charAt(0).toUpperCase() + pairData.trend.slice(1)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">Sentiment</span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      pairData.sentiment === 'positive' ? 'bg-green-100 text-green-800' :
                      pairData.sentiment === 'negative' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {pairData.sentiment.charAt(0).toUpperCase() + pairData.sentiment.slice(1)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Trading Recommendations */}
              <div>
                <h4 className="text-md font-semibold text-gray-900 mb-3">AI Recommendations</h4>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <div className="text-sm text-blue-800">
                    <p className="font-medium mb-2">Current Analysis:</p>
                    <p className="mb-2">
                      {pairData.trend === 'bullish' ? 
                        'Strong upward momentum detected. Consider long positions with proper risk management.' :
                        pairData.trend === 'bearish' ?
                        'Downward pressure observed. Consider short positions or wait for better entry.' :
                        'Sideways movement detected. Range trading opportunities may be present.'
                      }
                    </p>
                    <p className="text-xs text-blue-600">
                      RSI: {pairData.rsi > 70 ? 'Overbought conditions' : pairData.rsi < 30 ? 'Oversold conditions' : 'Neutral levels'} | 
                      MACD: {pairData.macd > 0 ? 'Bullish divergence' : 'Bearish divergence'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
