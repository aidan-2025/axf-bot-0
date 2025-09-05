import { useState } from 'react';
import Head from 'next/head';

export default function ClientTestPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testApi = async () => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Testing API calls...');
      
      // Test health endpoint
      const healthResponse = await fetch('http://localhost:8000/health');
      const healthData = await healthResponse.json();
      console.log('Health data:', healthData);
      
      // Test market data endpoint
      const marketResponse = await fetch('http://localhost:8000/api/v1/data/market');
      const marketData = await marketResponse.json();
      console.log('Market data:', marketData);
      
      setData({
        health: healthData,
        market: marketData
      });
    } catch (err) {
      console.error('Error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Client Test - AXF Bot 0</title>
      </Head>
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">Client-Side API Test</h1>
          
          <div className="mb-6">
            <button
              onClick={testApi}
              disabled={loading}
              className="bg-blue-600 text-white px-6 py-3 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Testing...' : 'Test API Calls'}
            </button>
          </div>

          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
              <strong>Error:</strong> {error}
            </div>
          )}

          {data && (
            <div className="space-y-6">
              <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4">Health Status</h2>
                <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
                  {JSON.stringify(data.health, null, 2)}
                </pre>
              </div>
              
              <div className="bg-white shadow rounded-lg p-6">
                <h2 className="text-xl font-semibold mb-4">Market Data</h2>
                <div className="mb-4">
                  <p className="text-sm text-gray-600">Currency Pairs: {data.market?.currencyPairs?.length || 0}</p>
                </div>
                <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto max-h-96">
                  {JSON.stringify(data.market, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
