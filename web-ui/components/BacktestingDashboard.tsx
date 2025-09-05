import React, { useState, useEffect } from 'react';
import { apiService, BacktestStrategy, BacktestResult, BacktestRanking, StrategyGenerationRequest, StrategyOptions } from '../lib/api';

interface BacktestingDashboardProps {
  className?: string;
}

const BacktestingDashboard: React.FC<BacktestingDashboardProps> = ({ className = '' }) => {
  const [strategies, setStrategies] = useState<BacktestStrategy[]>([]);
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([]);
  const [backtestResults, setBacktestResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [strategyOptions, setStrategyOptions] = useState<StrategyOptions | null>(null);
  const [showGenerationForm, setShowGenerationForm] = useState(false);
  const [currentStep, setCurrentStep] = useState<'intro' | 'generate' | 'select' | 'configure' | 'run' | 'results'>('intro');
  const [generationRequest, setGenerationRequest] = useState<StrategyGenerationRequest>({
    strategy_types: ['trend', 'range', 'breakout'],
    symbols: ['EURUSD', 'GBPUSD', 'USDJPY'],
    timeframes: ['1h', '4h', '1d'],
    count: 5,
    market_conditions: undefined,
    risk_level: undefined
  });

  // Initialize dates
  useEffect(() => {
    const today = new Date();
    const thirtyDaysAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);
    
    setStartDate(thirtyDaysAgo.toISOString().split('T')[0]);
    setEndDate(today.toISOString().split('T')[0]);
  }, []);

  // Load sample strategies and options
  useEffect(() => {
    const loadData = async () => {
      try {
        const [sampleResponse, optionsResponse] = await Promise.all([
          apiService.getSampleStrategies(),
          apiService.getStrategyOptions()
        ]);
        
        if (sampleResponse.success) {
          setStrategies(sampleResponse.strategies);
        }
        
        if (optionsResponse.success) {
          setStrategyOptions(optionsResponse.options);
        }
      } catch (err) {
        console.error('Failed to load data:', err);
        setError('Failed to load strategy data');
      }
    };

    loadData();
  }, []);

  const handleStrategyToggle = (strategyId: string) => {
    setSelectedStrategies(prev => 
      prev.includes(strategyId) 
        ? prev.filter(id => id !== strategyId)
        : [...prev, strategyId]
    );
  };

  const handleGenerateStrategies = async () => {
    setIsGenerating(true);
    setError(null);

    try {
      const response = await apiService.generateStrategies(generationRequest);
      if (response.success) {
        setStrategies(response.strategies);
        setShowGenerationForm(false);
        setSelectedStrategies([]);
        setCurrentStep('select');
      } else {
        setError(response.error || 'Failed to generate strategies');
      }
    } catch (err) {
      console.error('Strategy generation failed:', err);
      setError('Strategy generation failed. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunBacktest = async () => {
    if (selectedStrategies.length === 0) {
      setError('Please select at least one strategy');
      return;
    }

    if (!startDate || !endDate) {
      setError('Please select both start and end dates');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const selectedStrategyConfigs = strategies
        .filter(strategy => selectedStrategies.includes(strategy.strategy_id))
        .map(strategy => ({
          strategy_id: strategy.strategy_id,
          strategy_name: strategy.strategy_name,
          strategy_type: strategy.strategy_type,
          module_path: strategy.module_path,
          class_name: strategy.class_name,
          parameters: strategy.parameters
        }));

      const request = {
        strategies: selectedStrategyConfigs,
        start_date: new Date(startDate).toISOString(),
        end_date: new Date(endDate).toISOString(),
        max_workers: 4,
        timeout_seconds: 300
      };

      const response = await apiService.runBacktest(request);
      if (response.success) {
        setBacktestResults(response.results);
        setCurrentStep('results');
      } else {
        setError(response.error || 'Backtest failed');
      }
    } catch (err) {
      console.error('Backtest failed:', err);
      setError('Backtest failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const formatScore = (score: number) => (score * 100).toFixed(1);

  const renderIntroStep = () => (
    <div className="text-center py-12">
      <div className="mb-8">
        <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-4xl">🧪</span>
        </div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Strategy Backtesting Lab</h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Test and validate your trading strategies with historical data. Generate new strategies or use existing ones to find the best performing approaches.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="p-6 bg-blue-50 rounded-lg border border-blue-200">
          <div className="text-2xl mb-2">🎯</div>
          <h3 className="font-semibold text-gray-900 mb-2">Generate Strategies</h3>
          <p className="text-sm text-gray-600">Create new trading strategies with customizable parameters and risk profiles</p>
        </div>
        <div className="p-6 bg-green-50 rounded-lg border border-green-200">
          <div className="text-2xl mb-2">⚡</div>
          <h3 className="font-semibold text-gray-900 mb-2">Test Performance</h3>
          <p className="text-sm text-gray-600">Run backtests on historical data to validate strategy performance</p>
        </div>
        <div className="p-6 bg-purple-50 rounded-lg border border-purple-200">
          <div className="text-2xl mb-2">📊</div>
          <h3 className="font-semibold text-gray-900 mb-2">Analyze Results</h3>
          <p className="text-sm text-gray-600">Compare strategies with detailed metrics and rankings</p>
        </div>
      </div>

      <div className="space-x-4">
        <button
          onClick={() => setCurrentStep('generate')}
          className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
        >
          Start with Sample Strategies
        </button>
        <button
          onClick={() => setCurrentStep('generate')}
          className="px-8 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors"
        >
          Generate New Strategies
        </button>
      </div>
    </div>
  );

  const renderGenerateStep = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Generate Trading Strategies</h2>
        <p className="text-gray-600">Configure your strategy preferences and generate new trading strategies</p>
      </div>

      {strategyOptions && (
        <div className="p-6 bg-green-50 rounded-lg border border-green-200">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">Strategy Types</label>
              <div className="space-y-2">
                {strategyOptions.strategy_types.map(type => (
                  <label key={type} className="flex items-center p-2 hover:bg-green-100 rounded">
                    <input
                      type="checkbox"
                      checked={generationRequest.strategy_types.includes(type)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setGenerationRequest(prev => ({
                            ...prev,
                            strategy_types: [...prev.strategy_types, type]
                          }));
                        } else {
                          setGenerationRequest(prev => ({
                            ...prev,
                            strategy_types: prev.strategy_types.filter(t => t !== type)
                          }));
                        }
                      }}
                      className="mr-3"
                    />
                    <span className="text-sm capitalize font-medium">{type.replace('_', ' ')}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">Currency Pairs</label>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {strategyOptions.symbols.map(symbol => (
                  <label key={symbol} className="flex items-center p-2 hover:bg-green-100 rounded">
                    <input
                      type="checkbox"
                      checked={generationRequest.symbols.includes(symbol)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setGenerationRequest(prev => ({
                            ...prev,
                            symbols: [...prev.symbols, symbol]
                          }));
                        } else {
                          setGenerationRequest(prev => ({
                            ...prev,
                            symbols: prev.symbols.filter(s => s !== symbol)
                          }));
                        }
                      }}
                      className="mr-3"
                    />
                    <span className="text-sm font-medium">{symbol}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">Timeframes</label>
              <div className="space-y-2">
                {strategyOptions.timeframes.map(timeframe => (
                  <label key={timeframe} className="flex items-center p-2 hover:bg-green-100 rounded">
                    <input
                      type="checkbox"
                      checked={generationRequest.timeframes.includes(timeframe)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setGenerationRequest(prev => ({
                            ...prev,
                            timeframes: [...prev.timeframes, timeframe]
                          }));
                        } else {
                          setGenerationRequest(prev => ({
                            ...prev,
                            timeframes: prev.timeframes.filter(t => t !== timeframe)
                          }));
                        }
                      }}
                      className="mr-3"
                    />
                    <span className="text-sm font-medium">{timeframe}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Number of Strategies</label>
              <input
                type="number"
                min="1"
                max="20"
                value={generationRequest.count}
                onChange={(e) => setGenerationRequest(prev => ({
                  ...prev,
                  count: parseInt(e.target.value) || 5
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Market Conditions</label>
              <select
                value={generationRequest.market_conditions || ''}
                onChange={(e) => setGenerationRequest(prev => ({
                  ...prev,
                  market_conditions: e.target.value || undefined
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="">Any Market</option>
                {strategyOptions.market_conditions.map(condition => (
                  <option key={condition} value={condition} className="capitalize">
                    {condition}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Risk Level</label>
              <select
                value={generationRequest.risk_level || ''}
                onChange={(e) => setGenerationRequest(prev => ({
                  ...prev,
                  risk_level: e.target.value || undefined
                }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-green-500"
              >
                <option value="">Any Risk</option>
                {strategyOptions.risk_levels.map(level => (
                  <option key={level} value={level} className="capitalize">
                    {level}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-6 flex justify-center space-x-4">
            <button
              onClick={() => setCurrentStep('intro')}
              className="px-6 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Back
            </button>
            <button
              onClick={handleGenerateStrategies}
              disabled={isGenerating || generationRequest.strategy_types.length === 0 || generationRequest.symbols.length === 0 || generationRequest.timeframes.length === 0}
              className={`px-8 py-2 rounded-lg font-semibold transition-all ${
                isGenerating || generationRequest.strategy_types.length === 0 || generationRequest.symbols.length === 0 || generationRequest.timeframes.length === 0
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              {isGenerating ? 'Generating...' : `Generate ${generationRequest.count} Strategies`}
            </button>
          </div>
        </div>
      )}
    </div>
  );

  const renderSelectStep = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Select Strategies to Test</h2>
        <p className="text-gray-600">Choose which strategies you want to backtest</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {strategies.map(strategy => (
          <div
            key={strategy.strategy_id}
            className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
              selectedStrategies.includes(strategy.strategy_id)
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300'
            }`}
            onClick={() => handleStrategyToggle(strategy.strategy_id)}
          >
            <div className="flex items-start justify-between mb-2">
              <h3 className="font-semibold text-gray-900 text-sm">{strategy.strategy_name}</h3>
              <input
                type="checkbox"
                checked={selectedStrategies.includes(strategy.strategy_id)}
                onChange={() => handleStrategyToggle(strategy.strategy_id)}
                className="ml-2"
              />
            </div>
            <div className="space-y-1 text-xs text-gray-600">
              <div className="flex justify-between">
                <span>Type:</span>
                <span className="capitalize font-medium">{strategy.strategy_type}</span>
              </div>
              <div className="flex justify-between">
                <span>Symbol:</span>
                <span className="font-medium">{strategy.parameters.symbol}</span>
              </div>
              <div className="flex justify-between">
                <span>Timeframe:</span>
                <span className="font-medium">{strategy.parameters.timeframe}</span>
              </div>
              {strategy.market_conditions && (
                <div className="flex justify-between">
                  <span>Market:</span>
                  <span className="capitalize font-medium">{strategy.market_conditions}</span>
                </div>
              )}
              {strategy.risk_level && (
                <div className="flex justify-between">
                  <span>Risk:</span>
                  <span className="capitalize font-medium">{strategy.risk_level}</span>
                </div>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-2 line-clamp-2">{strategy.description}</p>
          </div>
        ))}
      </div>

      <div className="flex justify-center space-x-4">
        <button
          onClick={() => setCurrentStep('generate')}
          className="px-6 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={() => setCurrentStep('configure')}
          disabled={selectedStrategies.length === 0}
          className={`px-8 py-2 rounded-lg font-semibold transition-all ${
            selectedStrategies.length === 0
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          Configure Backtest ({selectedStrategies.length} selected)
        </button>
      </div>
    </div>
  );

  const renderConfigureStep = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Configure Backtest</h2>
        <p className="text-gray-600">Set the time period and parameters for your backtest</p>
      </div>

      <div className="max-w-2xl mx-auto">
        <div className="p-6 bg-gray-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Backtest Period</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
              <input
                type="date"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
              <input
                type="date"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        <div className="p-6 bg-blue-50 rounded-lg">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Selected Strategies</h3>
          <div className="space-y-2">
            {strategies
              .filter(s => selectedStrategies.includes(s.strategy_id))
              .map(strategy => (
                <div key={strategy.strategy_id} className="flex justify-between items-center p-2 bg-white rounded border">
                  <span className="font-medium text-sm">{strategy.strategy_name}</span>
                  <span className="text-xs text-gray-500">{strategy.parameters.symbol} {strategy.parameters.timeframe}</span>
                </div>
              ))}
          </div>
        </div>
      </div>

      <div className="flex justify-center space-x-4">
        <button
          onClick={() => setCurrentStep('select')}
          className="px-6 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={handleRunBacktest}
          disabled={isLoading || !startDate || !endDate}
          className={`px-8 py-2 rounded-lg font-semibold transition-all ${
            isLoading || !startDate || !endDate
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-green-600 text-white hover:bg-green-700'
          }`}
        >
          {isLoading ? 'Running Backtest...' : 'Run Backtest'}
        </button>
      </div>
    </div>
  );

  const renderResultsStep = () => (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Backtest Results</h2>
        <p className="text-gray-600">Analysis of your strategy performance</p>
      </div>

      {backtestResults && (
        <div className="space-y-6">
          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="text-2xl font-bold text-green-600">{backtestResults.summary.total_strategies}</div>
              <div className="text-sm text-gray-600">Strategies Tested</div>
            </div>
            <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{backtestResults.summary.successful_strategies}</div>
              <div className="text-sm text-gray-600">Successful</div>
            </div>
            <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">{formatScore(backtestResults.summary.average_score)}%</div>
              <div className="text-sm text-gray-600">Avg Score</div>
            </div>
            <div className="p-4 bg-orange-50 rounded-lg border border-orange-200">
              <div className="text-2xl font-bold text-orange-600">{backtestResults.summary.total_trades}</div>
              <div className="text-sm text-gray-600">Total Trades</div>
            </div>
          </div>

          {/* Strategy Rankings */}
          {backtestResults.rankings && backtestResults.rankings.length > 0 && (
            <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Strategy Rankings</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trades</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Win Rate</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sharpe</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {backtestResults.rankings.map((ranking, index) => (
                      <tr key={ranking.strategy_id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          #{ranking.rank}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">{ranking.strategy_name}</div>
                          <div className="text-sm text-gray-500">{ranking.strategy_type}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatScore(ranking.composite_score)}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {ranking.performance_metrics?.total_trades || 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {ranking.performance_metrics?.win_rate ? formatScore(ranking.performance_metrics.win_rate) + '%' : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {ranking.performance_metrics?.sharpe_ratio ? ranking.performance_metrics.sharpe_ratio.toFixed(2) : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="flex justify-center space-x-4">
        <button
          onClick={() => setCurrentStep('intro')}
          className="px-8 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
        >
          Start New Backtest
        </button>
        <button
          onClick={() => setCurrentStep('generate')}
          className="px-8 py-2 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 transition-colors"
        >
          Generate More Strategies
        </button>
      </div>
    </div>
  );

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      {/* Progress Indicator */}
      <div className="mb-8">
        <div className="flex items-center justify-center space-x-4">
          {[
            { key: 'intro', label: 'Start', icon: '🏁' },
            { key: 'generate', label: 'Generate', icon: '🎯' },
            { key: 'select', label: 'Select', icon: '✅' },
            { key: 'configure', label: 'Configure', icon: '⚙️' },
            { key: 'run', label: 'Run', icon: '🚀' },
            { key: 'results', label: 'Results', icon: '📊' }
          ].map((step, index) => (
            <div key={step.key} className="flex items-center">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-semibold ${
                currentStep === step.key
                  ? 'bg-blue-600 text-white'
                  : ['intro', 'generate', 'select', 'configure', 'run', 'results'].indexOf(currentStep) > index
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 text-gray-500'
              }`}>
                {['intro', 'generate', 'select', 'configure', 'run', 'results'].indexOf(currentStep) > index ? '✓' : step.icon}
              </div>
              <span className={`ml-2 text-sm font-medium ${
                currentStep === step.key ? 'text-blue-600' : 'text-gray-500'
              }`}>
                {step.label}
              </span>
              {index < 5 && (
                <div className={`w-8 h-0.5 mx-4 ${
                  ['intro', 'generate', 'select', 'configure', 'run', 'results'].indexOf(currentStep) > index
                    ? 'bg-green-500'
                    : 'bg-gray-200'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex">
            <div className="text-red-400 mr-3">⚠️</div>
            <div className="text-red-700">{error}</div>
          </div>
        </div>
      )}

      {/* Step Content */}
      {currentStep === 'intro' && renderIntroStep()}
      {currentStep === 'generate' && renderGenerateStep()}
      {currentStep === 'select' && renderSelectStep()}
      {currentStep === 'configure' && renderConfigureStep()}
      {currentStep === 'run' && renderConfigureStep()}
      {currentStep === 'results' && renderResultsStep()}
    </div>
  );
};

export default BacktestingDashboard;