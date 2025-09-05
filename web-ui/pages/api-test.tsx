import { useState, useEffect } from 'react';
import Head from 'next/head';

export default function ApiTestPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('Fetching data...');
        
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

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading API test...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-red-600 mb-4">API Test Failed</h1>
          <p className="text-gray-600">Error: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>API Test - AXF Bot 0</title>
      </Head>
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-900 mb-8">API Test Results</h1>
          
          <div className="space-y-6">
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Health Status</h2>
              <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto">
                {JSON.stringify(data?.health, null, 2)}
              </pre>
            </div>
            
            <div className="bg-white shadow rounded-lg p-6">
              <h2 className="text-xl font-semibold mb-4">Market Data</h2>
              <div className="mb-4">
                <p className="text-sm text-gray-600">Currency Pairs: {data?.market?.currencyPairs?.length || 0}</p>
              </div>
              <pre className="bg-gray-100 p-4 rounded text-sm overflow-auto max-h-96">
                {JSON.stringify(data?.market, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
