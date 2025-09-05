import React, { useState, useEffect } from 'react';

interface WorkflowConfig {
  strategy_count: number;
  strategy_types: string[];
  currency_pairs: string[];
  timeframes: string[];
  enable_parallel_processing: boolean;
  max_workers: number;
  batch_size: number;
  timeout_seconds: number;
  store_results: boolean;
  cleanup_old_results: boolean;
  days_to_keep: number;
  enable_logging: boolean;
  log_level: string;
  enable_metrics: boolean;
}

interface WorkflowStatus {
  workflow_id: string;
  status: string;
  message: string;
  current_stage?: string;
  progress_percentage: number;
  strategies_generated: number;
  strategies_validated: number;
  strategies_passed: number;
  strategies_stored: number;
  start_time?: string;
  duration: number;
  errors: string[];
  warnings: string[];
}

interface WorkflowResult {
  workflow_id: string;
  status: string;
  message: string;
  generated_strategies: any[];
  validation_results?: any;
  stored_results: any[];
  metrics: {
    start_time?: string;
    end_time?: string;
    duration: number;
    stage_timings: Record<string, number>;
    strategies_generated: number;
    strategies_validated: number;
    strategies_passed: number;
    strategies_failed: number;
    strategies_stored: number;
    strategies_per_second: number;
    average_validation_time: number;
    average_storage_time: number;
    total_errors: number;
    stage_errors: Record<string, number>;
    retry_count: number;
  };
  errors: string[];
  warnings: string[];
  created_at: string;
  updated_at: string;
}

const WorkflowDashboard: React.FC = () => {
  const [activeWorkflows, setActiveWorkflows] = useState<WorkflowStatus[]>([]);
  const [workflowHistory, setWorkflowHistory] = useState<WorkflowResult[]>([]);
  const [config, setConfig] = useState<WorkflowConfig>({
    strategy_count: 10,
    strategy_types: ['trend_following', 'mean_reversion', 'breakout'],
    currency_pairs: ['EURUSD', 'GBPUSD', 'USDJPY'],
    timeframes: ['H1', 'H4', 'D1'],
    enable_parallel_processing: true,
    max_workers: 4,
    batch_size: 5,
    timeout_seconds: 300,
    store_results: true,
    cleanup_old_results: true,
    days_to_keep: 30,
    enable_logging: true,
    log_level: 'INFO',
    enable_metrics: true
  });
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('active');

  // Fetch active workflows
  const fetchActiveWorkflows = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/workflow/active');
      if (response.ok) {
        const data = await response.json();
        setActiveWorkflows(data);
      }
    } catch (err) {
      console.error('Failed to fetch active workflows:', err);
    }
  };

  // Fetch workflow history
  const fetchWorkflowHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/workflow/history?limit=20');
      if (response.ok) {
        const data = await response.json();
        setWorkflowHistory(data);
      }
    } catch (err) {
      console.error('Failed to fetch workflow history:', err);
    }
  };

  // Start workflow
  const startWorkflow = async (sync = false) => {
    setIsRunning(true);
    setError(null);

    try {
      const endpoint = sync ? 'http://localhost:8000/api/v1/workflow/start-sync' : 'http://localhost:8000/api/v1/workflow/start';
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      if (response.ok) {
        const data = await response.json();
        if (sync) {
          // For sync workflow, add to history immediately
          await fetchWorkflowHistory();
        }
        // Refresh active workflows
        await fetchActiveWorkflows();
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'Failed to start workflow');
      }
    } catch (err) {
      setError('Network error: ' + (err as Error).message);
    } finally {
      setIsRunning(false);
    }
  };

  // Cancel workflow
  const cancelWorkflow = async (workflowId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/workflow/cancel/${workflowId}`, {
        method: 'POST',
      });

      if (response.ok) {
        await fetchActiveWorkflows();
      }
    } catch (err) {
      console.error('Failed to cancel workflow:', err);
    }
  };

  // Refresh data
  const refreshData = async () => {
    await Promise.all([fetchActiveWorkflows(), fetchWorkflowHistory()]);
  };

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const interval = setInterval(refreshData, 5000);
    return () => clearInterval(interval);
  }, []);

  // Initial load
  useEffect(() => {
    refreshData();
  }, []);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <span className="text-green-500">✅</span>;
      case 'failed':
        return <span className="text-red-500">❌</span>;
      case 'running':
        return <span className="text-blue-500 animate-spin">🔄</span>;
      case 'cancelled':
        return <span className="text-gray-500">⏹️</span>;
      default:
        return <span className="text-gray-500">⏰</span>;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Validation Workflow Dashboard</h1>
        <div className="flex space-x-2">
          <button 
            onClick={refreshData}
            className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50"
          >
            🔄 Refresh
          </button>
          <button 
            onClick={() => startWorkflow(false)} 
            disabled={isRunning}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            ▶️ Start Async
          </button>
          <button 
            onClick={() => startWorkflow(true)} 
            disabled={isRunning}
            className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            ▶️ Start Sync
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <span className="text-red-500 mr-2">❌</span>
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('active')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'active' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            🔄 Active Workflows ({activeWorkflows.length})
          </button>
          <button
            onClick={() => setActiveTab('history')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'history' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            📚 History ({workflowHistory.length})
          </button>
          <button
            onClick={() => setActiveTab('config')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'config' 
                ? 'border-blue-500 text-blue-600' 
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            ⚙️ Configuration
          </button>
        </nav>
      </div>

      {/* Active Workflows Tab */}
      {activeTab === 'active' && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium">Active Workflows</h3>
          </div>
          <div className="p-6">
            {activeWorkflows.length === 0 ? (
              <p className="text-gray-500">No active workflows</p>
            ) : (
              <div className="space-y-4">
                {activeWorkflows.map((workflow) => (
                  <div key={workflow.workflow_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(workflow.status)}
                        <div>
                          <p className="font-medium">{workflow.workflow_id}</p>
                          <p className="text-sm text-gray-500">{workflow.message}</p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(workflow.status)}`}>
                          {workflow.status}
                        </span>
                        {workflow.status === 'running' && (
                          <button
                            onClick={() => cancelWorkflow(workflow.workflow_id)}
                            className="px-3 py-1 text-xs border border-gray-300 rounded hover:bg-gray-50"
                          >
                            ⏹️ Cancel
                          </button>
                        )}
                      </div>
                    </div>
                    
                    {workflow.status === 'running' && (
                      <div className="mt-4">
                        <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                          <span>{workflow.current_stage || 'Initializing'}</span>
                          <span>{workflow.progress_percentage.toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${workflow.progress_percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    )}

                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Generated</p>
                        <p className="font-medium">{workflow.strategies_generated}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Validated</p>
                        <p className="font-medium">{workflow.strategies_validated}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Passed</p>
                        <p className="font-medium">{workflow.strategies_passed}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Stored</p>
                        <p className="font-medium">{workflow.strategies_stored}</p>
                      </div>
                    </div>

                    {workflow.errors.length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm text-red-600 font-medium">Errors:</p>
                        <ul className="text-sm text-red-600 list-disc list-inside">
                          {workflow.errors.map((error, index) => (
                            <li key={index}>{error}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {workflow.warnings.length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm text-yellow-600 font-medium">Warnings:</p>
                        <ul className="text-sm text-yellow-600 list-disc list-inside">
                          {workflow.warnings.map((warning, index) => (
                            <li key={index}>{warning}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium">Workflow History</h3>
          </div>
          <div className="p-6">
            {workflowHistory.length === 0 ? (
              <p className="text-gray-500">No workflow history</p>
            ) : (
              <div className="space-y-4">
                {workflowHistory.map((workflow) => (
                  <div key={workflow.workflow_id} className="border rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(workflow.status)}
                        <div>
                          <p className="font-medium">{workflow.workflow_id}</p>
                          <p className="text-sm text-gray-500">{workflow.message}</p>
                          <p className="text-xs text-gray-400">
                            {new Date(workflow.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(workflow.status)}`}>
                        {workflow.status}
                      </span>
                    </div>

                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-gray-500">Duration</p>
                        <p className="font-medium">{workflow.metrics.duration.toFixed(2)}s</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Generated</p>
                        <p className="font-medium">{workflow.metrics.strategies_generated}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Passed</p>
                        <p className="font-medium">{workflow.metrics.strategies_passed}</p>
                      </div>
                      <div>
                        <p className="text-gray-500">Stored</p>
                        <p className="font-medium">{workflow.metrics.strategies_stored}</p>
                      </div>
                    </div>

                    {workflow.metrics.strategies_per_second > 0 && (
                      <div className="mt-4 text-sm">
                        <p className="text-gray-500">Performance</p>
                        <p className="font-medium">
                          {workflow.metrics.strategies_per_second.toFixed(2)} strategies/second
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Configuration Tab */}
      {activeTab === 'config' && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium">Workflow Configuration</h3>
          </div>
          <div className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Strategy Count</label>
                  <input
                    type="number"
                    value={config.strategy_count}
                    onChange={(e) => setConfig({...config, strategy_count: parseInt(e.target.value)})}
                    className="w-full p-2 border rounded"
                    min="1"
                    max="100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Strategy Types</label>
                  <div className="space-y-2">
                    {['trend_following', 'mean_reversion', 'breakout', 'scalping'].map((type) => (
                      <label key={type} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={config.strategy_types.includes(type)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setConfig({...config, strategy_types: [...config.strategy_types, type]});
                            } else {
                              setConfig({...config, strategy_types: config.strategy_types.filter(t => t !== type)});
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm capitalize">{type.replace('_', ' ')}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Currency Pairs</label>
                  <div className="space-y-2">
                    {['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD'].map((pair) => (
                      <label key={pair} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={config.currency_pairs.includes(pair)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setConfig({...config, currency_pairs: [...config.currency_pairs, pair]});
                            } else {
                              setConfig({...config, currency_pairs: config.currency_pairs.filter(p => p !== pair)});
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm">{pair}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Timeframes</label>
                  <div className="space-y-2">
                    {['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'].map((tf) => (
                      <label key={tf} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={config.timeframes.includes(tf)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setConfig({...config, timeframes: [...config.timeframes, tf]});
                            } else {
                              setConfig({...config, timeframes: config.timeframes.filter(t => t !== tf)});
                            }
                          }}
                          className="mr-2"
                        />
                        <span className="text-sm">{tf}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Max Workers</label>
                  <input
                    type="number"
                    value={config.max_workers}
                    onChange={(e) => setConfig({...config, max_workers: parseInt(e.target.value)})}
                    className="w-full p-2 border rounded"
                    min="1"
                    max="16"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Timeout (seconds)</label>
                  <input
                    type="number"
                    value={config.timeout_seconds}
                    onChange={(e) => setConfig({...config, timeout_seconds: parseFloat(e.target.value)})}
                    className="w-full p-2 border rounded"
                    min="30"
                    max="3600"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default WorkflowDashboard;