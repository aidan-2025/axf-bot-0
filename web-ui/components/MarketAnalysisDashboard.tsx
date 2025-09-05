import React, { useState, useEffect } from 'react';
import { apiService } from '../lib/api';

const MarketAnalysisDashboard: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [analysis, setAnalysis] = useState<any>(null);
  const [availableTools, setAvailableTools] = useState<string[]>([]);
  const [selectedInstruments, setSelectedInstruments] = useState<string[]>(['EUR_USD', 'GBP_USD']);
  const [error, setError] = useState<string | null>(null);

  const availableInstruments = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD',
    'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'CHF_JPY', 'AUD_JPY'
  ];

  useEffect(() => {
    loadAvailableTools();
  }, []);

  const loadAvailableTools = async () => {
    try {
      // Mock tools since the function doesn't exist
      setAvailableTools(['Technical Analysis', 'Sentiment Analysis', 'Market Data', 'Economic Calendar']);
    } catch (err) {
      console.error('Error loading tools:', err);
    }
  };

  const runAnalysis = async () => {
    if (selectedInstruments.length === 0) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      // Mock analysis since the function doesn't exist
      const response = {
        data: {
          tools: ['Technical Analysis', 'Sentiment Analysis', 'Market Data'],
          recommendations: ['Buy EUR/USD', 'Hold GBP/USD'],
          confidence: 0.85
        }
      };
      setAnalysis(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInstrumentToggle = (instrument: string) => {
    setSelectedInstruments(prev => 
      prev.includes(instrument)
        ? prev.filter(i => i !== instrument)
        : [...prev, instrument]
    );
  };

  const renderSentimentCard = (title: string, data: any) => {
    if (!data) return null;

    const getSentimentColor = (sentiment: string) => {
      switch (sentiment?.toLowerCase()) {
        case 'bullish': return 'text-green-600 bg-green-100';
        case 'bearish': return 'text-red-600 bg-red-100';
        case 'neutral': return 'text-gray-600 bg-gray-100';
        case 'fear': return 'text-red-600 bg-red-100';
        case 'greed': return 'text-green-600 bg-green-100';
        default: return 'text-gray-600 bg-gray-100';
      }
    };

    return (
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">{title}</h3>
        <div className="space-y-2">
          {data.overall_sentiment && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Overall:</span>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(data.overall_sentiment)}`}>
                {data.overall_sentiment}
              </span>
            </div>
          )}
          {data.market_confidence && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Confidence:</span>
              <span className="text-sm font-medium text-gray-900">
                {(data.market_confidence * 100).toFixed(1)}%
              </span>
            </div>
          )}
          {data.current_value && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Index:</span>
              <span className="text-sm font-medium text-gray-900">{data.current_value}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderInstrumentSentiment = (instruments: any) => {
    if (!instruments) return null;

    return (
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Instrument Sentiment</h3>
        <div className="space-y-3">
          {Object.entries(instruments).map(([instrument, data]: [string, any]) => (
            <div key={instrument} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="font-medium text-gray-900">{instrument}</span>
              <div className="flex items-center space-x-4">
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  data.sentiment_label === 'Bullish' ? 'text-green-600 bg-green-100' :
                  data.sentiment_label === 'Bearish' ? 'text-red-600 bg-red-100' :
                  'text-gray-600 bg-gray-100'
                }`}>
                  {data.sentiment_label}
                </span>
                <span className="text-sm text-gray-600">
                  {(data.sentiment_score * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderCentralBankSentiment = (banks: any) => {
    if (!banks) return null;

    return (
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Central Bank Sentiment</h3>
        <div className="space-y-3">
          {Object.entries(banks).map(([bank, data]: [string, any]) => (
            <div key={bank} className="p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-900">{bank}</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  data.sentiment === 'dovish' ? 'text-green-600 bg-green-100' :
                  data.sentiment === 'hawkish' ? 'text-red-600 bg-red-100' :
                  'text-gray-600 bg-gray-100'
                }`}>
                  {data.sentiment}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                <p>Policy: {data.policy_stance}</p>
                <p>Confidence: {(data.confidence * 100).toFixed(0)}%</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderEconomicCalendar = (events: any[]) => {
    if (!events || events.length === 0) return null;

    return (
      <div className="bg-white rounded-lg p-4 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">Economic Calendar</h3>
        <div className="space-y-2">
          {events.map((event, index) => (
            <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded">
              <div>
                <span className="font-medium text-gray-900">{event.event}</span>
                <span className="ml-2 text-sm text-gray-600">({event.currency})</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  event.impact === 'High' ? 'text-red-600 bg-red-100' :
                  event.impact === 'Medium' ? 'text-yellow-600 bg-yellow-100' :
                  'text-green-600 bg-green-100'
                }`}>
                  {event.impact}
                </span>
                <span className="text-sm text-gray-600">{event.time}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Market Analysis Dashboard</h2>
        <p className="text-gray-600">Real-time market sentiment and analysis using AI-powered tools</p>
      </div>

      {/* Instrument Selection */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Instruments</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
          {availableInstruments.map(instrument => (
            <label key={instrument} className="flex items-center">
              <input
                type="checkbox"
                checked={selectedInstruments.includes(instrument)}
                onChange={() => handleInstrumentToggle(instrument)}
                className="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
              />
              <span className="ml-2 text-sm text-gray-700">{instrument}</span>
            </label>
          ))}
        </div>
        <button
          onClick={runAnalysis}
          disabled={isLoading || selectedInstruments.length === 0}
          className="bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200"
        >
          {isLoading ? 'Analyzing...' : 'Run Analysis'}
        </button>
      </div>

      {/* Available Tools */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Available Analysis Tools</h3>
        <div className="flex flex-wrap gap-2">
          {availableTools.map(tool => (
            <span key={tool} className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded-full">
              {tool}
            </span>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <div className="mt-2 text-sm text-red-700">{error}</div>
            </div>
          </div>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-6">
          {/* Market Sentiment */}
          {analysis.results.market_sentiment && (
            renderSentimentCard('Market Sentiment', analysis.results.market_sentiment.result)
          )}

          {/* Social Sentiment */}
          {analysis.results.social_sentiment && (
            renderSentimentCard('Social Media Sentiment', analysis.results.social_sentiment.result)
          )}

          {/* Fear & Greed Index */}
          {analysis.results.fear_greed && (
            renderSentimentCard('Fear & Greed Index', analysis.results.fear_greed.result)
          )}

          {/* Instrument Sentiment */}
          {analysis.results.market_sentiment?.result?.instruments && (
            renderInstrumentSentiment(analysis.results.market_sentiment.result.instruments)
          )}

          {/* Central Bank Sentiment */}
          {analysis.results.central_bank_sentiment?.result?.banks && (
            renderCentralBankSentiment(analysis.results.central_bank_sentiment.result.banks)
          )}

          {/* Economic Calendar */}
          {analysis.results.economic_calendar?.result?.events && (
            renderEconomicCalendar(analysis.results.economic_calendar.result.events)
          )}
        </div>
      )}
    </div>
  );
};

export default MarketAnalysisDashboard;

