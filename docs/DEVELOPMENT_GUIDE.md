# AXF Bot 0 - Development Guide

This comprehensive guide covers everything you need to know to develop, test, and contribute to the AXF Bot 0 project.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Development Environment Setup](#development-environment-setup)
4. [Project Structure](#project-structure)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Database Management](#database-management)
8. [API Development](#api-development)
9. [Frontend Development](#frontend-development)
10. [Docker Development](#docker-development)
11. [Monitoring and Debugging](#monitoring-and-debugging)
12. [Deployment](#deployment)
13. [Contributing](#contributing)
14. [Troubleshooting](#troubleshooting)

## Project Overview

AXF Bot 0 is a comprehensive AI-powered Forex trading system consisting of:

- **App1**: AI-Powered Forex Strategy Generator (FastAPI + Python)
- **App2**: MetaTrader 4 Script Development Application (FastAPI + Python)
- **Web UI**: Next.js dashboard for strategy management and monitoring
- **Database**: PostgreSQL for data storage and Redis for caching
- **Monitoring**: Prometheus + Grafana for system monitoring

### Key Features

- Real-time market data processing
- AI-powered strategy generation
- Sentiment analysis from news
- Strategy performance monitoring
- MetaTrader 4 Expert Advisor generation
- Comprehensive web dashboard
- Real-time monitoring and alerting

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/aidan-2025/axf-bot-0.git
cd axf-bot-0

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Create environment file
make secrets-create

# Validate configuration
make secrets-validate
```

### 3. Start Development Environment

```bash
# Start all services
make dev-start

# Check status
make health-check
```

### 4. Access Services

- **Web UI**: http://localhost:3000
- **App1 API**: http://localhost:8000
- **App2 API**: http://localhost:8001
- **Database Admin**: http://localhost:5050
- **Redis Admin**: http://localhost:8002
- **Grafana**: http://localhost:3001

## Development Environment Setup

### Local Development (Recommended)

1. **Python Environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r app1/requirements.txt
   pip install -r app2/requirements.txt
   ```

2. **Node.js Environment**:
   ```bash
   cd web-ui
   npm install
   ```

3. **Database Setup**:
   ```bash
   # Start database services
   docker-compose -f docker-compose.dev.yml up -d postgres redis influxdb
   
   # Run migrations
   make migrate
   
   # Seed database
   make seed-db
   ```

### Docker Development

1. **Full Docker Environment**:
   ```bash
   # Start all services
   make dev-start
   
   # View logs
   make dev-logs
   
   # Stop services
   make dev-stop
   ```

2. **Individual Services**:
   ```bash
   # Start specific service
   docker-compose -f docker-compose.dev.yml up -d app1
   
   # View service logs
   docker-compose -f docker-compose.dev.yml logs -f app1
   ```

## Project Structure

```
axf-bot-0/
├── app1/                          # Strategy Generator API
│   ├── src/
│   │   ├── api/                   # FastAPI routes
│   │   ├── data_ingestion/        # Market data processing
│   │   ├── sentiment_analysis/    # News sentiment analysis
│   │   ├── strategy_generation/   # AI strategy generation
│   │   ├── strategy_monitoring/   # Performance tracking
│   │   └── database/              # Database models and connections
│   ├── tests/                     # Unit and integration tests
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Multi-stage Docker build
├── app2/                          # MT4 EA Generator API
│   ├── src/
│   │   ├── api/                   # FastAPI routes
│   │   ├── ea_generation/         # Expert Advisor generation
│   │   ├── backtesting/           # Strategy backtesting
│   │   └── database/              # Database models
│   ├── tests/                     # Unit and integration tests
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Multi-stage Docker build
├── web-ui/                        # Next.js Dashboard
│   ├── components/                # React components
│   ├── pages/                     # Next.js pages
│   ├── lib/                       # Utility functions
│   ├── styles/                    # CSS and styling
│   ├── package.json               # Node.js dependencies
│   └── Dockerfile                 # Multi-stage Docker build
├── config/                        # Configuration files
│   └── environment.py             # Environment management
├── deployment/                    # Deployment configurations
│   ├── prometheus.yml             # Prometheus configuration
│   ├── alert_rules.yml            # Alert rules
│   └── grafana/                   # Grafana dashboards
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
│   ├── setup.sh                   # Initial setup
│   ├── dev-workflow.sh            # Development commands
│   ├── monitor.sh                 # Monitoring tools
│   ├── manage-secrets.sh          # Secrets management
│   └── migrate.py                 # Database migrations
├── tests/                         # Integration tests
├── docker-compose.yml             # Production Docker Compose
├── docker-compose.dev.yml         # Development Docker Compose
├── docker-compose.secrets.yml     # Production with secrets
├── Makefile                       # Development commands
└── README.md                      # Project overview
```

## Development Workflow

### 1. Branch Strategy

We use Git Flow branching strategy:

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation

### 2. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add your feature"

# Push and create PR
git push origin feature/your-feature-name
```

### 3. Code Quality

```bash
# Run linting
make lint

# Run tests
make test

# Run type checking
make type-check

# Run security scan
make security-scan
```

### 4. Database Changes

```bash
# Create migration
python scripts/migrate.py --create "migration_name"

# Apply migrations
make migrate

# Rollback migration
python scripts/migrate.py --rollback
```

## Testing

### Unit Tests

```bash
# Run all tests
make test

# Run specific app tests
cd app1 && python -m pytest
cd app2 && python -m pytest

# Run with coverage
make test-coverage
```

### Integration Tests

```bash
# Run integration tests
make test-integration

# Test specific service
make test-app1
make test-app2
make test-web-ui
```

### API Testing

```bash
# Test App1 API
curl http://localhost:8000/health

# Test App2 API
curl http://localhost:8001/health

# Test Web UI
curl http://localhost:3000/api/health
```

## Database Management

### Migrations

```bash
# Create new migration
python scripts/migrate.py --create "add_new_table"

# Apply migrations
make migrate

# Rollback last migration
python scripts/migrate.py --rollback

# Show migration status
python scripts/migrate.py --status
```

### Database Access

```bash
# Connect to database
docker-compose -f docker-compose.dev.yml exec postgres psql -U postgres -d axf_bot_db

# Backup database
make db-backup

# Restore database
make db-restore backup_file.sql
```

### Seeding Data

```bash
# Seed with sample data
make seed-db

# Clear and reseed
make db-reset && make seed-db
```

## API Development

### App1 - Strategy Generator

**Key Endpoints**:
- `GET /health` - Health check
- `GET /api/v1/strategies` - List strategies
- `POST /api/v1/strategies` - Create strategy
- `GET /api/v1/performance` - Performance metrics

**Development**:
```bash
# Start App1 in development mode
cd app1
python -m uvicorn main:app --reload --port 8000

# Run tests
python -m pytest tests/
```

### App2 - MT4 EA Generator

**Key Endpoints**:
- `GET /health` - Health check
- `POST /api/v1/ea/generate` - Generate EA
- `POST /api/v1/ea/backtest` - Backtest strategy

**Development**:
```bash
# Start App2 in development mode
cd app2
python -m uvicorn main:app --reload --port 8001

# Run tests
python -m pytest tests/
```

## Frontend Development

### Web UI Development

```bash
# Start development server
cd web-ui
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run linting
npm run lint
```

### Component Development

```bash
# Create new component
mkdir web-ui/components/NewComponent
touch web-ui/components/NewComponent/index.tsx
touch web-ui/components/NewComponent/NewComponent.module.css
```

## Docker Development

### Development Containers

```bash
# Start development environment
make dev-start

# View logs
make dev-logs

# Access container shell
make dev-shell

# Restart services
make dev-restart
```

### Building Images

```bash
# Build all images
make build

# Build specific service
docker-compose -f docker-compose.dev.yml build app1

# Build with no cache
docker-compose -f docker-compose.dev.yml build --no-cache app1
```

## Monitoring and Debugging

### Health Checks

```bash
# Check all services
make health-check

# Check specific service
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:3000/api/health
```

### Monitoring

```bash
# View system overview
make monitor

# Watch system status
make watch

# View detailed metrics
./scripts/monitor.sh detailed
```

### Logging

```bash
# View all logs
make dev-logs

# View specific service logs
docker-compose -f docker-compose.dev.yml logs -f app1
docker-compose -f docker-compose.dev.yml logs -f app2
docker-compose -f docker-compose.dev.yml logs -f web-ui
```

### Debugging

```bash
# Debug App1
docker-compose -f docker-compose.dev.yml exec app1 python -m debugpy --listen 0.0.0.0:5678 --wait-for-client main.py

# Debug App2
docker-compose -f docker-compose.dev.yml exec app2 python -m debugpy --listen 0.0.0.0:5679 --wait-for-client main.py
```

## Deployment

### Production Deployment

```bash
# Build production images
make build-prod

# Deploy with secrets
docker-compose -f docker-compose.secrets.yml up -d

# Check deployment status
make health-check
```

### Staging Deployment

```bash
# Deploy to staging
make deploy-staging

# Run staging tests
make test-staging
```

## Contributing

### Code Standards

1. **Python**: Follow PEP 8, use type hints, write docstrings
2. **JavaScript/TypeScript**: Use ESLint, Prettier, write JSDoc
3. **Commits**: Use conventional commit messages
4. **Tests**: Write tests for new features
5. **Documentation**: Update docs for new features

### Pull Request Process

1. Create feature branch from `develop`
2. Make changes and write tests
3. Run all checks: `make lint test type-check`
4. Create pull request
5. Address review feedback
6. Merge after approval

### Code Review Checklist

- [ ] Code follows project standards
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Error handling is proper

## Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check port usage
lsof -i :8000
lsof -i :8001
lsof -i :3000

# Kill process using port
kill -9 $(lsof -t -i:8000)
```

#### 2. Database Connection Issues
```bash
# Check database status
docker-compose -f docker-compose.dev.yml exec postgres pg_isready -U postgres

# Reset database
make db-reset
```

#### 3. Docker Issues
```bash
# Clean Docker system
docker system prune -a

# Rebuild containers
make dev-stop
make dev-start
```

#### 4. Environment Issues
```bash
# Check environment status
make secrets-status

# Recreate environment
rm .env
make secrets-create
```

### Getting Help

1. Check this documentation
2. Check the troubleshooting section
3. Check GitHub issues
4. Ask in team chat
5. Create new issue if needed

### Useful Commands

```bash
# Quick status check
make status

# View all available commands
make help

# Clean everything and start fresh
make clean && make dev-start

# View system resources
docker stats

# Check disk usage
df -h
```

## Additional Resources

- [API Documentation](http://localhost:8000/docs) - App1 API docs
- [API Documentation](http://localhost:8001/docs) - App2 API docs
- [Environment Setup Guide](docs/ENVIRONMENT_SETUP.md)
- [Database Schema](docs/DATABASE_SCHEMA.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Happy Coding! 🚀**

For questions or issues, please check the troubleshooting section or create an issue on GitHub.
