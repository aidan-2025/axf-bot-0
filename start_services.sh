#!/bin/bash

# AXF Bot 0 - Service Startup Script
# This script starts all services with proper health checks

set -e  # Exit on any error

echo "🚀 Starting AXF Bot 0 Services..."
echo "=================================="

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "❌ Port $port is already in use"
        return 1
    else
        echo "✅ Port $port is available"
        return 0
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Kill any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f uvicorn 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
sleep 2

# Check ports
echo "🔍 Checking port availability..."
check_port 8000 || exit 1
check_port 3000 || exit 1

# Start Docker services
echo "🐳 Starting Docker services..."
cd /Users/alextsoi/Projects/axf-bot-0
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting Docker containers..."
    docker-compose up -d postgres redis influxdb
    sleep 5
else
    echo "Docker containers already running"
fi

# Verify Docker services
echo "🔍 Verifying Docker services..."
if ! docker-compose ps | grep -q "postgres.*Up"; then
    echo "❌ PostgreSQL container not running"
    exit 1
fi
echo "✅ PostgreSQL is running"

# Start FastAPI backend
echo "🔧 Starting FastAPI backend..."
cd /Users/alextsoi/Projects/axf-bot-0/app1
source ../.venv/bin/activate

# Test database connection first
echo "🔍 Testing database connection..."
python -c "
from src.database.connection import check_db_connection, get_database_status
print('Database status:', get_database_status())
if check_db_connection():
    print('✅ Database connection successful')
else:
    print('❌ Database connection failed')
    exit(1)
"

# Start uvicorn in background
echo "🚀 Starting uvicorn server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!

# Wait for backend to be ready
wait_for_service "http://localhost:8000/health" "FastAPI Backend" || exit 1

# Test API endpoints
echo "🧪 Testing API endpoints..."
curl -s "http://localhost:8000/api/v1/strategies/" >/dev/null && echo "✅ Strategies endpoint working" || echo "❌ Strategies endpoint failed"
curl -s "http://localhost:8000/api/v1/data/market" >/dev/null && echo "✅ Market data endpoint working" || echo "❌ Market data endpoint failed"
curl -s "http://localhost:8000/api/v1/performance/summary" >/dev/null && echo "✅ Performance endpoint working" || echo "❌ Performance endpoint failed"

# Start Next.js frontend
echo "🎨 Starting Next.js frontend..."
cd /Users/alextsoi/Projects/axf-bot-0/web-ui
npm run dev &
NEXT_PID=$!

# Wait for frontend to be ready
wait_for_service "http://localhost:3000" "Next.js Frontend" || exit 1

# Test frontend
echo "🧪 Testing frontend..."
curl -s "http://localhost:3000" | grep -q "AXF Bot" && echo "✅ Frontend loading correctly" || echo "❌ Frontend not loading"

echo ""
echo "🎉 All services started successfully!"
echo "=================================="
echo "📊 Backend API: http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $UVICORN_PID 2>/dev/null || true
    kill $NEXT_PID 2>/dev/null || true
    pkill -f uvicorn 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
wait
