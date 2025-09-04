# Environment Setup and Configuration

This document explains how to set up and manage environment variables and secrets for the AXF Bot 0 project.

## Overview

The project uses a centralized environment management system that supports:
- Development and production configurations
- Secure secrets management
- Docker secrets integration
- Environment validation
- Automatic secret generation

## Quick Start

### 1. Create Environment File

```bash
# Create .env file from development template
make secrets-create

# Or manually
./scripts/manage-secrets.sh create development
```

### 2. Validate Configuration

```bash
# Validate environment variables
make secrets-validate

# Or manually
./scripts/manage-secrets.sh validate .env
```

### 3. Check Status

```bash
# Show environment status
make secrets-status

# Or manually
./scripts/manage-secrets.sh status .env
```

## Environment Files

### Development (`env.development`)
- Debug mode enabled
- Local database connections
- Development API keys
- Hot reloading enabled

### Production (`env.production`)
- Debug mode disabled
- Containerized database connections
- Production API keys
- Optimized for performance

## Configuration Structure

### Database Configuration
```bash
DATABASE_URL=postgresql://user:password@host:port/database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=axf_bot_db
DB_USER=postgres
DB_PASSWORD=password
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

### Redis Configuration
```bash
REDIS_URL=redis://host:port/db
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
```

### InfluxDB Configuration
```bash
INFLUXDB_URL=http://host:port
INFLUXDB_HOST=localhost
INFLUXDB_PORT=8086
INFLUXDB_TOKEN=
INFLUXDB_ORG=axf-bot
INFLUXDB_BUCKET=trading-data
```

### API Keys
```bash
FANO_API_KEY=your_fano_api_key_here
FANO_BASE_URL=https://api.fano.ai
NEWS_API_KEY=your_news_api_key_here
ECONOMIC_CALENDAR_API_KEY=your_economic_calendar_api_key_here
```

### Security Configuration
```bash
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

## Secrets Management

### Docker Secrets (Production)

1. **Create secrets directory**:
   ```bash
   mkdir -p secrets
   ```

2. **Generate secret files**:
   ```bash
   # Generate secret key
   ./scripts/manage-secrets.sh generate-key > secrets/secret_key.txt
   
   # Generate database password
   openssl rand -base64 32 > secrets/db_password.txt
   
   # Generate Redis password
   openssl rand -base64 32 > secrets/redis_password.txt
   
   # Generate InfluxDB password
   openssl rand -base64 32 > secrets/influxdb_password.txt
   
   # Add your API keys
   echo "your_fano_api_key_here" > secrets/fano_api_key.txt
   echo "your_news_api_key_here" > secrets/news_api_key.txt
   ```

3. **Set proper permissions**:
   ```bash
   chmod 600 secrets/*.txt
   ```

4. **Use with Docker Compose**:
   ```bash
   docker-compose -f docker-compose.secrets.yml up -d
   ```

### Environment File Encryption

1. **Encrypt sensitive values**:
   ```bash
   make secrets-encrypt
   ```

2. **Decrypt when needed**:
   ```bash
   ./scripts/manage-secrets.sh decrypt .env.encrypted .env
   ```

## Available Commands

### Make Commands
```bash
# Secrets Management
make secrets-create      # Create .env file from template
make secrets-validate    # Validate environment variables
make secrets-encrypt     # Encrypt sensitive values
make secrets-status      # Show environment status

# Development
make dev-start          # Start development environment
make dev-stop           # Stop development environment
make dev-restart        # Restart development environment
make dev-logs           # View development logs
make dev-shell          # Open shell in app1 container

# Database
make db-setup           # Set up database
make db-reset           # Reset database (WARNING: deletes data)
make migrate            # Run database migrations
make seed-db            # Seed database with sample data

# Monitoring
make health-check       # Check system health
make monitor            # Show system monitoring
make watch              # Watch system status continuously
```

### Script Commands
```bash
# Secrets Management
./scripts/manage-secrets.sh create [type]     # Create .env file
./scripts/manage-secrets.sh validate [file]   # Validate environment
./scripts/manage-secrets.sh encrypt [file]    # Encrypt secrets
./scripts/manage-secrets.sh decrypt [file]    # Decrypt secrets
./scripts/manage-secrets.sh status [file]     # Show status
./scripts/manage-secrets.sh generate-key      # Generate secret key

# Monitoring
./scripts/monitor.sh overview                 # System overview
./scripts/monitor.sh detailed                 # Detailed monitoring
./scripts/monitor.sh health                   # Health checks
./scripts/monitor.sh resources                # Resource usage
./scripts/monitor.sh logs [service]           # Service logs
./scripts/monitor.sh watch                    # Watch status
```

## Security Best Practices

### 1. Never Commit Secrets
- Add `secrets/*.txt` to `.gitignore`
- Use environment-specific files
- Use Docker secrets in production

### 2. Use Strong Passwords
- Minimum 32 characters
- Mix of letters, numbers, symbols
- Different passwords for each environment

### 3. Rotate Secrets Regularly
- Change passwords monthly
- Update API keys when possible
- Monitor access logs

### 4. Restrict File Permissions
```bash
chmod 600 secrets/*.txt
chmod 600 .env
```

### 5. Use Environment-Specific Secrets
- Development: Use test keys
- Staging: Use staging keys
- Production: Use production keys

## Troubleshooting

### Common Issues

#### 1. Permission Denied
```bash
chmod 600 secrets/*.txt
chmod 600 .env
```

#### 2. Secret Not Found
```bash
# Check if secret file exists
ls -la secrets/

# Check Docker secret
docker secret ls
```

#### 3. Invalid Secret Format
```bash
# Check secret content (first few characters)
head -c 10 secrets/secret_key.txt
```

#### 4. Environment Variable Not Set
```bash
# Check environment status
make secrets-status

# Validate environment
make secrets-validate
```

### Debug Commands

```bash
# Show all environment variables
env | grep -E "(DB_|REDIS_|INFLUXDB_|FANO_|NEWS_)"

# Check Docker container environment
docker-compose -f docker-compose.dev.yml exec app1 env

# Test database connection
docker-compose -f docker-compose.dev.yml exec postgres pg_isready -U postgres

# Test Redis connection
docker-compose -f docker-compose.dev.yml exec redis redis-cli ping
```

## Environment Variables Reference

### Required Variables
- `DATABASE_URL` - Database connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - Application secret key

### Optional Variables
- `DEBUG` - Enable debug mode (default: false)
- `LOG_LEVEL` - Logging level (default: INFO)
- `HOST` - Application host (default: 0.0.0.0)
- `PORT` - Application port (default: 8000)

### API Keys
- `FANO_API_KEY` - Fano STT API key
- `NEWS_API_KEY` - News API key
- `ECONOMIC_CALENDAR_API_KEY` - Economic calendar API key

### Database
- `DB_HOST` - Database host
- `DB_PORT` - Database port
- `DB_NAME` - Database name
- `DB_USER` - Database user
- `DB_PASSWORD` - Database password
- `DB_POOL_SIZE` - Connection pool size
- `DB_MAX_OVERFLOW` - Maximum overflow connections

### Redis
- `REDIS_HOST` - Redis host
- `REDIS_PORT` - Redis port
- `REDIS_PASSWORD` - Redis password
- `REDIS_DB` - Redis database number

### InfluxDB
- `INFLUXDB_HOST` - InfluxDB host
- `INFLUXDB_PORT` - InfluxDB port
- `INFLUXDB_TOKEN` - InfluxDB token
- `INFLUXDB_ORG` - InfluxDB organization
- `INFLUXDB_BUCKET` - InfluxDB bucket

### Security
- `ALGORITHM` - JWT algorithm
- `ACCESS_TOKEN_EXPIRE_MINUTES` - Access token expiration
- `REFRESH_TOKEN_EXPIRE_DAYS` - Refresh token expiration

### Monitoring
- `PROMETHEUS_ENABLED` - Enable Prometheus
- `GRAFANA_ENABLED` - Enable Grafana
- `METRICS_INTERVAL` - Metrics collection interval

## Support

For issues with environment setup:
1. Check the troubleshooting section
2. Run `make secrets-status` to see current configuration
3. Run `make secrets-validate` to check for errors
4. Check the logs with `make dev-logs`
