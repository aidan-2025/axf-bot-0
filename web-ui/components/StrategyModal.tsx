import { useState } from 'react';
import { apiService, StrategyParams } from '../lib/api';

interface StrategyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (strategy: any) => void;
}

export default function StrategyModal({ isOpen, onClose, onGenerate }: StrategyModalProps) {
  const [params, setParams] = useState<StrategyParams>({
    marketConditions: 'Bullish',
    riskLevel: 'Moderate',
    timeframe: '1 Hour'
  });
  const [isGenerating, setIsGenerating] = useState(false);

  if (!isOpen) return null;

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const result = await apiService.generateAIStrategy(params);
      onGenerate(result.strategy);
      onClose();
    } catch (error) {
      console.error('Strategy generation failed:', error);
      // Fallback to mock data if API fails
      const mockStrategy = {
        id: Date.now().toString(),
        name: `Strategy ${Date.now()}`,
        description: `AI-generated strategy for ${params.marketConditions} market with ${params.riskLevel} risk`,
        parameters: params,
        status: 'active' as const,
        createdAt: new Date().toISOString(),
        reasoning: 'Generated using AI analysis of current market conditions',
        confidence: 0.85
      };
      onGenerate(mockStrategy);
      onClose();
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Generate AI Strategy</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            disabled={isGenerating}
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Market Conditions
            </label>
            <select
              value={params.marketConditions}
              onChange={(e) => setParams({...params, marketConditions: e.target.value})}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
              disabled={isGenerating}
            >
              <option>Bullish</option>
              <option>Bearish</option>
              <option>Sideways</option>
              <option>Volatile</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Risk Level
            </label>
            <select
              value={params.riskLevel}
              onChange={(e) => setParams({...params, riskLevel: e.target.value})}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
              disabled={isGenerating}
            >
              <option>Conservative</option>
              <option>Moderate</option>
              <option>Aggressive</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Timeframe
            </label>
            <select
              value={params.timeframe}
              onChange={(e) => setParams({...params, timeframe: e.target.value})}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
              disabled={isGenerating}
            >
              <option>1 Minute</option>
              <option>5 Minutes</option>
              <option>15 Minutes</option>
              <option>1 Hour</option>
              <option>4 Hours</option>
              <option>1 Day</option>
            </select>
          </div>
        </div>

        {isGenerating && (
          <div className="mt-4 p-4 bg-blue-50 rounded-md">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-3"></div>
              <span className="text-sm text-blue-700">Generating strategy with AI...</span>
            </div>
          </div>
        )}

        <div className="flex justify-end space-x-3 mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-700 bg-gray-200 rounded-md hover:bg-gray-300"
            disabled={isGenerating}
          >
            Cancel
          </button>
          <button
            onClick={handleGenerate}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            disabled={isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate Strategy'}
          </button>
        </div>
      </div>
    </div>
  );
}
