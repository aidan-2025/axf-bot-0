import React, { useState } from 'react';
import { apiService } from '../lib/api';

interface AIStrategyGeneratorProps {
  onStrategyGenerated?: (strategy: any) => void;
}

const AIStrategyGenerator: React.FC<AIStrategyGeneratorProps> = ({ onStrategyGenerated }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<string>('');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  const [formData, setFormData] = useState<any>({
    instruments: ['EUR_USD', 'GBP_USD'],
    timeframes: ['M15', 'H1'],
    risk: 'medium'
  });

  const availableInstruments = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'USD_CHF', 'AUD_USD', 'USD_CAD',
    'NZD_USD', 'EUR_GBP', 'EUR_JPY', 'GBP_JPY', 'CHF_JPY', 'AUD_JPY'
  ];

  const availableTimeframes = [
    'M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1'
  ];

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    
    try {
      setCurrentStep('Analyzing market conditions...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setCurrentStep('Generating strategy parameters...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setCurrentStep('Optimizing strategy...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setCurrentStep('Finalizing strategy...');
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Simulate a generated strategy
      const mockStrategy = {
        id: `strategy_${Date.now()}`,
        name: `AI Strategy for ${formData.instruments.join(', ')}`,
        description: `AI-generated strategy optimized for ${formData.timeframes.join(', ')} timeframes with ${formData.risk} risk level`,
        parameters: formData,
        status: 'testing' as const,
        createdAt: new Date().toISOString(),
        performance: {
          totalReturn: Math.random() * 20 - 5, // Random between -5% and 15%
          sharpeRatio: Math.random() * 2 + 0.5, // Random between 0.5 and 2.5
          maxDrawdown: Math.random() * 10 + 2 // Random between 2% and 12%
        }
      };
      
      setResult(mockStrategy);
      setCurrentStep('Strategy generated successfully!');
      
      if (onStrategyGenerated) {
        onStrategyGenerated(mockStrategy);
      }
      
    } catch (err) {
      setError('Failed to generate strategy. Please try again.');
      setCurrentStep('');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold mb-4">AI Strategy Generator</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Currency Pairs
          </label>
          <div className="grid grid-cols-3 gap-2">
            {availableInstruments.map(instrument => (
              <label key={instrument} className="flex items-center">
                <input
                  type="checkbox"
                  checked={formData.instruments.includes(instrument)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      handleInputChange('instruments', [...formData.instruments, instrument]);
                    } else {
                      handleInputChange('instruments', formData.instruments.filter((i: string) => i !== instrument));
                    }
                  }}
                  className="mr-2"
                />
                <span className="text-sm">{instrument}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Timeframes
          </label>
          <div className="grid grid-cols-4 gap-2">
            {availableTimeframes.map(timeframe => (
              <label key={timeframe} className="flex items-center">
                <input
                  type="checkbox"
                  checked={formData.timeframes.includes(timeframe)}
                  onChange={(e) => {
                    if (e.target.checked) {
                      handleInputChange('timeframes', [...formData.timeframes, timeframe]);
                    } else {
                      handleInputChange('timeframes', formData.timeframes.filter((t: string) => t !== timeframe));
                    }
                  }}
                  className="mr-2"
                />
                <span className="text-sm">{timeframe}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Risk Level
          </label>
          <select
            value={formData.risk}
            onChange={(e) => handleInputChange('risk', e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md"
          >
            <option value="low">Low Risk</option>
            <option value="medium">Medium Risk</option>
            <option value="high">High Risk</option>
          </select>
        </div>

        <button
          onClick={handleGenerate}
          disabled={isLoading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Generating...' : 'Generate AI Strategy'}
        </button>

        {currentStep && (
          <div className="text-sm text-blue-600">
            {currentStep}
          </div>
        )}

        {error && (
          <div className="text-sm text-red-600 bg-red-50 p-3 rounded-md">
            {error}
          </div>
        )}

        {result && (
          <div className="mt-4 p-4 bg-green-50 rounded-md">
            <h4 className="font-medium text-green-800 mb-2">Generated Strategy</h4>
            <div className="text-sm text-green-700">
              <p><strong>Name:</strong> {result.name}</p>
              <p><strong>Description:</strong> {result.description}</p>
              <p><strong>Status:</strong> {result.status}</p>
              <p><strong>Total Return:</strong> {result.performance.totalReturn.toFixed(2)}%</p>
              <p><strong>Sharpe Ratio:</strong> {result.performance.sharpeRatio.toFixed(2)}</p>
              <p><strong>Max Drawdown:</strong> {result.performance.maxDrawdown.toFixed(2)}%</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIStrategyGenerator;