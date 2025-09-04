# Quick Start Guide

Get up and running with AXF Bot 0 in minutes!

## 🚀 5-Minute Setup

### Prerequisites

- Docker and Docker Compose
- Git
- 8GB+ RAM

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/aidan-2025/axf-bot-0.git
cd axf-bot-0

# Create environment file
make secrets-create

# Start development environment
make dev-start
```

### Step 2: Verify Installation

```bash
# Check system health
make health-check

# View system status
make monitor
```

### Step 3: Access Services

- **Web Dashboard**: http://localhost:3000
- **Strategy API**: http://localhost:8000
- **EA Generator API**: http://localhost:8001
- **Database Admin**: http://localhost:5050
- **Monitoring**: http://localhost:3001

## 🎯 What You Can Do

### 1. Generate Trading Strategies

```bash
# Create a new strategy
curl -X POST http://localhost:8000/api/v1/strategies \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Strategy",
    "strategy_type": "trend_following",
    "parameters": {
      "timeframe": "1H",
      "indicators": ["rsi", "macd"]
    }
  }'
```

### 2. Generate MetaTrader 4 Expert Advisor

```bash
# Generate EA from strategy
curl -X POST http://localhost:8001/api/v1/ea/generate \
  -H "Content-Type: application/json" \
  -d '{
    "strategy_id": 1,
    "ea_name": "MyEA",
    "version": "1.0"
  }'
```

### 3. Monitor Performance

```bash
# Check strategy performance
curl http://localhost:8000/api/v1/performance/1
```

## 🛠️ Common Commands

```bash
# Start development environment
make dev-start

# Stop development environment
make dev-stop

# View logs
make dev-logs

# Check health
make health-check

# Monitor system
make monitor

# Access database
make db-shell

# Run migrations
make migrate

# Seed database
make seed-db
```

## 🔧 Configuration

### Environment Variables

```bash
# View current configuration
make secrets-status

# Validate configuration
make secrets-validate

# Update configuration
# Edit .env file
```

### API Keys

Add your API keys to `.env`:

```bash
FANO_API_KEY=your_fano_api_key_here
NEWS_API_KEY=your_news_api_key_here
ECONOMIC_CALENDAR_API_KEY=your_economic_calendar_api_key_here
```

## 📊 Monitoring

### Health Checks

```bash
# Check all services
make health-check

# Check specific service
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:3000/api/health
```

### System Monitoring

```bash
# View system overview
make monitor

# Watch system status
make watch

# View detailed metrics
./scripts/monitor.sh detailed
```

## 🐛 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
lsof -i :8000
lsof -i :8001
lsof -i :3000

# Kill process using port
kill -9 $(lsof -t -i:8000)
```

#### Database Issues
```bash
# Check database status
docker-compose -f docker-compose.dev.yml exec postgres pg_isready -U postgres

# Reset database
make db-reset
```

#### Docker Issues
```bash
# Clean Docker system
docker system prune -a

# Rebuild containers
make dev-stop
make dev-start
```

### Debug Commands

```bash
# View all logs
make dev-logs

# View specific service logs
docker-compose -f docker-compose.dev.yml logs -f app1

# Check system resources
docker stats

# Check disk usage
df -h
```

## 📚 Next Steps

1. **Read the [Development Guide](DEVELOPMENT_GUIDE.md)** for detailed development information
2. **Check the [Database Schema](DATABASE_SCHEMA.md)** for data structure details
3. **Review the [Deployment Guide](DEPLOYMENT.md)** for production deployment
4. **Explore the [Environment Setup](ENVIRONMENT_SETUP.md)** for configuration details

## 🆘 Need Help?

- Check the [troubleshooting section](#-troubleshooting)
- View the [full documentation](docs/)
- Create an [issue on GitHub](https://github.com/aidan-2025/axf-bot-0/issues)
- Ask in the team chat

## 🎉 Success!

You're now ready to start developing with AXF Bot 0! 

- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Monitoring**: http://localhost:3001

Happy coding! 🚀
