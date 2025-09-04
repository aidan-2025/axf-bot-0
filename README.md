# AXF Bot 0 - AI-Powered Forex Trading System

A comprehensive AI-powered forex trading system consisting of two interconnected applications that generate profitable trading strategies and convert them into executable MetaTrader 4 Expert Advisors.

## 🚀 Project Overview

The axf-bot-0 project aims to develop a comprehensive AI-powered forex trading system with two main components:

1. **AI-Powered Forex Strategy Generator** - Analyzes market data, news sentiment, and economic events to generate profitable trading strategies
2. **MetaTrader 4 Script Development Application** - Converts generated strategies into executable MT4 Expert Advisors with built-in self-evaluation capabilities

## 🏗️ Architecture

```
axf-bot-0/
├── app1/                          # AI-Powered Forex Strategy Generator
│   ├── src/
│   │   ├── data_ingestion/        # Market data collection and processing
│   │   ├── strategy_generation/   # AI strategy generation algorithms
│   │   ├── sentiment_analysis/    # News sentiment processing
│   │   ├── backtesting/          # Strategy validation and testing
│   │   └── api/                  # REST API endpoints
│   ├── config/                   # Configuration settings
│   ├── data/                     # Market data storage
│   ├── models/                   # ML model storage
│   └── tests/                    # Unit and integration tests
├── app2/                          # MetaTrader 4 Script Development
│   ├── src/
│   │   ├── code_generation/      # MQL4 code generation engine
│   │   ├── self_evaluation/      # Performance monitoring
│   │   ├── fault_detection/      # Strategy fault detection
│   │   └── mt4_integration/      # MT4 platform integration
│   ├── templates/                # MQL4 code templates
│   ├── generated/                # Generated EA files
│   └── backtesting/              # Backtesting results
├── .taskmaster/                  # Task management system
├── docs/                         # Documentation
├── scripts/                      # Utility scripts
└── deployment/                   # Docker and deployment configs
```

## 🛠️ Technology Stack

### Application 1 (Strategy Generator)
- **Language**: Python 3.11+
- **ML/AI**: TensorFlow, PyTorch, scikit-learn, transformers
- **Data Processing**: Pandas, NumPy, pandas-ta, TA-Lib
- **API**: FastAPI, uvicorn
- **Database**: PostgreSQL, InfluxDB, Redis
- **News/Sentiment**: newspaper3k, textblob, vaderSentiment

### Application 2 (MT4 EA Generator)
- **Language**: Python 3.11+
- **Code Generation**: Jinja2, Mako templates
- **API**: FastAPI, uvicorn
- **Database**: PostgreSQL
- **File Processing**: PyYAML, JSON

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Version Control**: Git
- **Task Management**: Taskmaster AI

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- Git

### 1. Clone the Repository
```bash
git clone <repository-url>
cd axf-bot-0
```

### 2. Set Up Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
# Copy example environment file
cp env.example .env

# Edit .env with your API keys and configuration
nano .env
```

### 4. Initialize Taskmaster
```bash
# Initialize the project (already done)
task-master init

# Parse the PRD to generate tasks
task-master parse-prd .taskmaster/docs/prd.md --num-tasks 25 --research
```

### 5. Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 6. Run Applications Locally
```bash
# Terminal 1 - Strategy Generator
cd app1
python main.py

# Terminal 2 - MT4 EA Generator
cd app2
python main.py
```

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

## 🎯 Success Metrics

- **Strategy Success Rate**: >70% profitable strategies in backtesting
- **Generation Speed**: <10 minutes per strategy
- **Maximum Drawdown**: <15% during major economic events
- **Code Generation**: <5 minutes from strategy to complete EA
- **System Uptime**: 99.5% availability during trading hours

## 📈 Development Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [x] Project initialization and structure
- [ ] Basic data ingestion for major currency pairs
- [ ] Database schema and API framework
- [ ] Simple EA generator with moving average strategy

### Phase 2: Core Features (Weeks 5-8)
- [ ] Sentiment analysis engine
- [ ] Forex Factory calendar integration
- [ ] Strategy generation algorithms
- [ ] Self-evaluation framework

### Phase 3: Advanced Features (Weeks 9-12)
- [ ] Multi-timeframe analysis
- [ ] Advanced ML models
- [ ] Risk management during events
- [ ] Strategy optimization

### Phase 4: Integration & Testing (Weeks 13-16)
- [ ] Real-time communication between apps
- [ ] Unified monitoring system
- [ ] Comprehensive testing
- [ ] Production deployment

## 🔧 Configuration

### Environment Variables
Key environment variables to configure:

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
pytest

# Run with coverage
pytest --cov=app1 --cov=app2

# Run specific app tests
pytest app1/tests/
pytest app2/tests/
```

## 📚 Documentation

- [API Documentation](docs/api/)
- [Strategy Generation Guide](docs/strategy-generation/)
- [MT4 Integration Guide](docs/mt4-integration/)
- [Deployment Guide](docs/deployment/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Risk Disclaimer

This software is for educational and research purposes only. Forex trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in the `docs/` folder
- Review the task management system in `.taskmaster/`

---

**Built with ❤️ using Taskmaster AI for intelligent project management**
