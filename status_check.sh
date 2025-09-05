#!/bin/bash

echo "🔍 AXF Bot 0 - Comprehensive Status Check"
echo "========================================="

# Check if services are running
echo "📊 Service Status:"
echo "------------------"

# Check FastAPI backend
if curl -s "http://localhost:8000/health" >/dev/null 2>&1; then
    echo "✅ FastAPI Backend: Running on port 8000"
else
    echo "❌ FastAPI Backend: Not responding"
fi

# Check Next.js frontend
if curl -s "http://localhost:3000" >/dev/null 2>&1; then
    echo "✅ Next.js Frontend: Running on port 3000"
else
    echo "❌ Next.js Frontend: Not responding"
fi

# Check Docker services
echo ""
echo "🐳 Docker Services:"
echo "------------------"
if docker-compose ps | grep -q "postgres.*Up"; then
    echo "✅ PostgreSQL: Running"
else
    echo "❌ PostgreSQL: Not running"
fi

if docker-compose ps | grep -q "redis.*Up"; then
    echo "✅ Redis: Running"
else
    echo "❌ Redis: Not running"
fi

if docker-compose ps | grep -q "influxdb.*Up"; then
    echo "✅ InfluxDB: Running"
else
    echo "❌ InfluxDB: Not running"
fi

# Test API endpoints
echo ""
echo "🧪 API Endpoint Tests:"
echo "---------------------"

# Test strategies endpoint
if curl -s "http://localhost:8000/api/v1/strategies/" | jq -e '.data.all | length > 0' >/dev/null 2>&1; then
    STRATEGIES_COUNT=$(curl -s "http://localhost:8000/api/v1/strategies/" | jq '.data.all | length')
    echo "✅ Strategies API: $STRATEGIES_COUNT strategies loaded"
else
    echo "❌ Strategies API: Failed or no data"
fi

# Test market data endpoint
if curl -s "http://localhost:8000/api/v1/data/market" | jq -e 'length > 0' >/dev/null 2>&1; then
    MARKET_COUNT=$(curl -s "http://localhost:8000/api/v1/data/market" | jq 'length')
    echo "✅ Market Data API: $MARKET_COUNT pairs loaded"
else
    echo "❌ Market Data API: Failed or no data"
fi

# Test performance endpoint
if curl -s "http://localhost:8000/api/v1/performance/summary" | jq -e '.data' >/dev/null 2>&1; then
    echo "✅ Performance API: Data available"
else
    echo "❌ Performance API: Failed or no data"
fi

# Test pair analysis endpoint
if curl -s "http://localhost:8000/api/v1/data/analyze/EURUSD" | jq -e '.symbol' >/dev/null 2>&1; then
    echo "✅ Pair Analysis API: Working"
else
    echo "❌ Pair Analysis API: Failed or not implemented"
fi

# Test frontend data loading
echo ""
echo "🎨 Frontend Data Loading:"
echo "------------------------"
if curl -s "http://localhost:3000" | grep -q "Portfolio Value"; then
    echo "✅ Dashboard: Loading correctly"
else
    echo "❌ Dashboard: Not loading properly"
fi

if curl -s "http://localhost:3000" | grep -q "Active Strategies"; then
    echo "✅ Strategies Display: Working"
else
    echo "❌ Strategies Display: Not working"
fi

# Market data display is client-side rendered, so we check if the page loads correctly
if curl -s "http://localhost:3000" | grep -q "Total Pairs"; then
    echo "✅ Market Data Display: Working (client-side rendered)"
else
    echo "❌ Market Data Display: Not working"
fi

echo ""
echo "📋 Summary:"
echo "----------"
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "All systems should be operational if you see ✅ for all checks above."
