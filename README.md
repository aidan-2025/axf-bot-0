# AXF Bot 0 - AI-Powered Forex Trading System

[![CI](https://github.com/alextsoi/axf-bot-0/actions/workflows/ci.yml/badge.svg)](https://github.com/alextsoi/axf-bot-0/actions/workflows/ci.yml)
[![CD](https://github.com/alextsoi/axf-bot-0/actions/workflows/cd.yml/badge.svg)](https://github.com/alextsoi/axf-bot-0/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 18+](https://img.shields.io/badge/node.js-18+-green.svg)](https://nodejs.org/)

A comprehensive AI-powered forex trading system consisting of two interconnected applications that generate profitable trading strategies and convert them into executable MetaTrader 4 Expert Advisors.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/alextsoi/axf-bot-0.git
cd axf-bot-0
```

### 2. Set Up Development Environment
```bash
# Run the setup script
./scripts/setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd web-ui && npm install
```

### 3. Start the Applications
```bash
# Using Docker (Recommended)
docker-compose up -d

# Or start individually
make start-docker
```

### 4. Access the Applications
- **Web UI Dashboard**: http://localhost:3000
- **App1 (Strategy Generator)**: http://localhost:8000
- **App2 (MT4 EA Generator)**: http://localhost:8001
- **Database**: localhost:5432

## 🏗️ Architecture

```
axf-bot-0/
├── app1/                          # AI-Powered Forex Strategy Generator
│   ├── src/
│   │   ├── data_ingestion/        # Market data collection and processing
│   │   ├── strategy_generation/   # AI strategy generation algorithms
│   │   ├── sentiment_analysis/    # News sentiment processing
│   │   ├── backtesting/          # Strategy validation and testing
│   │   ├── api/                  # REST API endpoints
│   │   └── strategy_monitoring/   # Performance tracking
│   ├── config/                   # Configuration settings
│   └── tests/                    # Unit and integration tests
├── app2/                          # MetaTrader 4 Script Development
│   ├── src/
│   │   ├── code_generation/      # MQL4 code generation engine
│   │   ├── self_evaluation/      # Performance monitoring
│   │   ├── fault_detection/      # Strategy fault detection
│   │   └── mt4_integration/      # MT4 platform integration
│   ├── templates/                # MQL4 code templates
│   └── generated/                # Generated EA files
├── web-ui/                        # Next.js Web Dashboard
│   ├── components/               # React components
│   ├── pages/                    # Next.js pages
│   └── lib/                      # API client and utilities
├── .taskmaster/                  # Task management system
├── docs/                         # Documentation
└── scripts/                      # Utility scripts
```

## 🛠️ Technology Stack

### Backend (Python)
- **FastAPI** - Modern, fast web framework
- **PostgreSQL** - Primary database
- **Redis** - Caching and session storage
- **InfluxDB** - Time series data storage
- **TensorFlow/PyTorch** - Machine learning models
- **Pandas/NumPy** - Data processing

### Frontend (Next.js)
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **React Query** - Data fetching

### Infrastructure
- **Docker** - Containerization
- **GitHub Actions** - CI/CD
- **Prometheus/Grafana** - Monitoring

## 📊 Key Features

### Strategy Generator (App1)
- ✅ Real-time market data integration (28+ currency pairs)
- ✅ Multi-timeframe analysis (M1 to W1)
- ✅ News sentiment analysis with NLP
- ✅ Economic calendar integration
- ✅ AI-powered strategy generation
- ✅ Comprehensive backtesting with real spreads
- ✅ Risk management during economic events

### MT4 EA Generator (App2)
- ✅ Automatic MQL4 code generation
- ✅ Universal variable system
- ✅ Self-evaluation and performance monitoring
- ✅ Fault detection and recovery
- ✅ Automated backtesting integration
- ✅ Strategy summary reports

### Web Dashboard
- ✅ Real-time market insights
- ✅ Strategy performance monitoring
- ✅ Interactive charts and visualizations
- ✅ Strategy management interface
- ✅ Performance analytics and reporting

## 🎯 Success Metrics

- **Strategy Success Rate**: >70% profitable strategies in backtesting
- **Generation Speed**: <10 minutes per strategy
- **Maximum Drawdown**: <15% during major economic events
- **Code Generation**: <5 minutes from strategy to complete EA
- **System Uptime**: 99.5% availability during trading hours

## 🚀 Development

### Available Commands
```bash
# Development
make dev              # Set up development environment
make test             # Run all tests
make lint             # Run linting checks
make format           # Format code

# Applications
make start-app1       # Start Strategy Generator
make start-app2       # Start MT4 EA Generator
make start-docker     # Start all services with Docker

# Database
make db-setup         # Set up database
make db-reset         # Reset database (WARNING: deletes data)

# Task Management
make tasks            # Show Taskmaster tasks
make next-task        # Show next task to work on
```

### Branching Strategy
We follow [Git Flow](https://danielkummer.github.io/git-flow-cheatsheet/):
- **`main`** - Production-ready code
- **`develop`** - Integration branch for features
- **`feature/*`** - New features and enhancements
- **`hotfix/*`** - Critical production fixes

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Performance Monitoring

### Strategy Performance Tracking
- **Profit Factor**: Ratio of gross profit to gross loss
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Performance Score**: 0-100 scale based on multiple metrics

### Well-Performing Strategy Detection
- Automatically identifies strategies performing >60% score
- Alerts on strategies that need attention (<60% score)
- Historical performance tracking
- Current market condition evaluation

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://username:password@localhost:5432/axf_bot_db
REDIS_URL=redis://localhost:6379/0
INFLUXDB_URL=http://localhost:8086

# API Keys
REUTERS_API_KEY=your_reuters_api_key
BLOOMBERG_API_KEY=your_bloomberg_api_key
FOREX_FACTORY_API_KEY=your_forex_factory_api_key

# Application Settings
APP1_PORT=8000
APP2_PORT=8001
DEBUG=True
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific tests
pytest app1/tests/
pytest app2/tests/
cd web-ui && npm test

# Run with coverage
pytest --cov=app1 --cov=app2 --cov-report=html
```

## 📚 Documentation

- [Quick Start Guide](docs/QUICK_START.md) - Get up and running in 5 minutes
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Comprehensive development information
- [Database Schema](docs/DATABASE_SCHEMA.md) - Database structure and relationships
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Environment Setup](docs/ENVIRONMENT_SETUP.md) - Configuration and secrets management
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute to the project
- [Setup Guide](SETUP_UI.md) - Complete setup instructions
- [Branching Strategy](docs/BRANCHING_STRATEGY.md) - Git workflow

## 🔍 Monitoring & Health Checks

### Health Endpoints
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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Risk Disclaimer

This software is for educational and research purposes only. Forex trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/alextsoi/axf-bot-0/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alextsoi/axf-bot-0/discussions)
- **Documentation**: [Project Wiki](https://github.com/alextsoi/axf-bot-0/wiki)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📊 Project Status

- **Total Tasks**: 25
- **Completed**: 1
- **In Progress**: 0
- **Next Task**: Provision Development and Testing Infrastructure

---

**Built with ❤️ using Taskmaster AI for intelligent project management**