import { useState, useEffect } from 'react';
import { apiService } from '../lib/api';

export default function TestData() {
  const [marketData, setMarketData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const data = await apiService.getMarketData();
        setMarketData(data);
      } catch (err) {
        console.error('Error loading market data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
        
        // Fallback data
        setMarketData([
          {
            symbol: "EUR/USD",
            price: 1.08542,
            change24h: 0.00123,
            changePercent24h: 0.11,
            volume: 1234567,
            high24h: 1.08750,
            low24h: 1.08320,
            timestamp: new Date().toISOString()
          },
          {
            symbol: "GBP/USD",
            price: 1.26478,
            change24h: -0.00234,
            changePercent24h: -0.18,
            volume: 987654,
            high24h: 1.26890,
            low24h: 1.26210,
            timestamp: new Date().toISOString()
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return <div>Loading market data...</div>;
  }

  return (
    <div>
      <h1>Market Data Test</h1>
      {error && <div style={{color: 'red'}}>Error: {error}</div>}
      <div>
        <h2>Market Data ({marketData.length} pairs):</h2>
        {marketData.map((pair, index) => (
          <div key={index} style={{border: '1px solid #ccc', margin: '10px', padding: '10px'}}>
            <h3>{pair.symbol}</h3>
            <p>Price: {pair.price}</p>
            <p>24h Change: {pair.change24h} ({pair.changePercent24h}%)</p>
            <p>Volume: {pair.volume.toLocaleString()}</p>
            <p>High: {pair.high24h} | Low: {pair.low24h}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
