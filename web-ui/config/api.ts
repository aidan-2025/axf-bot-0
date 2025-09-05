export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
};

export const ENDPOINTS = {
  HEALTH: '/health',
  MARKET_DATA: '/api/v1/data/market',
  STRATEGIES: '/api/v1/strategies/',
  STRATEGY_GENERATE: '/api/v1/ai/strategies/generate',
  STRATEGY_STREAM: '/api/v1/ai/strategies/stream',
  PAIR_ANALYSIS: '/api/v1/ai/analysis/pair',
  PERFORMANCE: '/api/v1/performance/summary',
} as const;
