# Web UI Setup Guide

## 🎯 Quick Start

### 1. Start All Services
```bash
# Using Docker Compose (Recommended)
docker-compose up -d

# Or start individually
make start-docker
```

### 2. Access the Applications
- **Web UI Dashboard**: http://localhost:3000
- **App1 (Strategy Generator)**: http://localhost:8000
- **App2 (MT4 EA Generator)**: http://localhost:8001
- **Database**: localhost:5432

### 3. Web UI Features
- 📊 **Dashboard**: Overview of all strategies and performance
- 📈 **Market Insights**: Real-time currency pair data and sentiment
- 🎯 **Strategy Management**: View, filter, and manage strategies
- 📊 **Performance Metrics**: Detailed performance analysis and charts
- 🔍 **Strategy Monitoring**: Check which past strategies are performing well

## 🗄️ Database Schema (Simplified)

### Main Tables:
1. **strategies** - Stores all generated strategies with parameters
2. **strategy_performance** - Daily performance tracking
3. **expert_advisors** - Generated MT4 EAs

### Key Features:
- JSON parameters for flexibility
- Simple performance metrics
- Automatic timestamp tracking
- Sample data included

## 🔧 Development

### Web UI Development
```bash
cd web-ui
npm install
npm run dev
```

### API Development
```bash
# App1 (Strategy Generator)
cd app1
python main.py

# App2 (MT4 EA Generator)  
cd app2
python main.py
```

### Database Setup
```bash
# Reset database with sample data
make db-reset

# Or setup fresh
make db-setup
```

## 📊 Performance Monitoring

### Check Well-Performing Strategies
```bash
curl http://localhost:8000/api/v1/performance/well-performing
```

### Get Performance Summary
```bash
curl http://localhost:8000/api/v1/performance/summary
```

### Check Specific Strategy
```bash
curl http://localhost:8000/api/v1/performance/strategy/STRAT_001
```

## 🎨 UI Components

### Dashboard Sections:
- **Overview**: Key metrics and recent strategies
- **Strategies**: Full strategy management interface
- **Performance**: Detailed performance analysis
- **Market Insights**: Real-time market data and sentiment

### Key Features:
- Real-time data updates
- Interactive charts and graphs
- Strategy filtering and sorting
- Performance scoring system
- Responsive design

## 🚀 Production Deployment

### Environment Variables
```bash
# Web UI
NEXT_PUBLIC_APP1_API_URL=http://your-app1-url
NEXT_PUBLIC_APP2_API_URL=http://your-app2-url

# Apps
DATABASE_URL=postgresql://user:pass@host:port/db
```

### Docker Production
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🔍 Monitoring & Health Checks

### Health Endpoints:
- Web UI: http://localhost:3000/api/health
- App1: http://localhost:8000/health
- App2: http://localhost:8001/health

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f web-ui
docker-compose logs -f app1
docker-compose logs -f app2
```

## 📈 Strategy Performance Features

### What the System Tracks:
- ✅ **Profit Factor**: Ratio of gross profit to gross loss
- ✅ **Win Rate**: Percentage of profitable trades
- ✅ **Max Drawdown**: Maximum peak-to-trough decline
- ✅ **Sharpe Ratio**: Risk-adjusted return measure
- ✅ **Total Trades**: Number of trades executed
- ✅ **Total Profit**: Cumulative profit/loss

### Performance Scoring:
- **0-100 scale** based on multiple metrics
- **>60% = Well Performing** strategies
- **<60% = Needs Attention** strategies
- **Automatic evaluation** against current market conditions

### Strategy Comparison:
- Side-by-side performance comparison
- Historical performance tracking
- Market condition analysis
- Risk-adjusted returns

## 🛠️ Troubleshooting

### Common Issues:
1. **Port conflicts**: Ensure ports 3000, 8000, 8001, 5432 are available
2. **Database connection**: Check PostgreSQL is running
3. **API errors**: Verify both apps are running
4. **Build errors**: Check Node.js and Python versions

### Reset Everything:
```bash
docker-compose down -v
docker-compose up -d
```

---

**🎉 Your AXF Bot 0 system is now ready with a complete web interface!**
