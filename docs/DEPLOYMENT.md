# Deployment Guide

This guide covers deploying the AXF Bot 0 system to various environments, from development to production.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Development Deployment](#development-deployment)
5. [Staging Deployment](#staging-deployment)
6. [Production Deployment](#production-deployment)
7. [Docker Deployment](#docker-deployment)
8. [Kubernetes Deployment](#kubernetes-deployment)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)

## Overview

The AXF Bot 0 system supports multiple deployment strategies:

- **Development**: Local Docker Compose for development
- **Staging**: Cloud-based staging environment
- **Production**: High-availability production deployment
- **Docker**: Containerized deployment
- **Kubernetes**: Orchestrated container deployment

## Prerequisites

### System Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB+ (16GB+ recommended for production)
- **Storage**: 100GB+ SSD
- **Network**: Stable internet connection

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Git
- Python 3.11+ (for local development)
- Node.js 18+ (for local development)

### Cloud Requirements (for cloud deployment)

- Cloud provider account (AWS, GCP, Azure)
- Domain name (for production)
- SSL certificate
- Load balancer
- Database service (RDS, Cloud SQL, etc.)

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/aidan-2025/axf-bot-0.git
cd axf-bot-0
```

### 2. Environment Configuration

```bash
# Create environment file
make secrets-create

# Validate configuration
make secrets-validate

# Check status
make secrets-status
```

### 3. Install Dependencies

```bash
# Python dependencies
pip install -r requirements.txt
pip install -r app1/requirements.txt
pip install -r app2/requirements.txt

# Node.js dependencies
cd web-ui && npm install && cd ..
```

## Development Deployment

### Local Development

```bash
# Start development environment
make dev-start

# Check status
make health-check

# View logs
make dev-logs
```

### Services

- **Web UI**: http://localhost:3000
- **App1 API**: http://localhost:8000
- **App2 API**: http://localhost:8001
- **Database Admin**: http://localhost:5050
- **Redis Admin**: http://localhost:8002
- **Grafana**: http://localhost:3001

### Development Commands

```bash
# Start services
make dev-start

# Stop services
make dev-stop

# Restart services
make dev-restart

# View logs
make dev-logs

# Access shell
make dev-shell

# Run migrations
make migrate

# Seed database
make seed-db

# Health check
make health-check

# Monitor system
make monitor
```

## Staging Deployment

### 1. Prepare Staging Environment

```bash
# Create staging branch
git checkout -b staging

# Update environment for staging
cp env.production .env
# Edit .env with staging values

# Build staging images
make build-staging
```

### 2. Deploy to Staging

```bash
# Deploy with staging configuration
docker-compose -f docker-compose.staging.yml up -d

# Check deployment
make health-check

# Run staging tests
make test-staging
```

### 3. Staging Configuration

```yaml
# docker-compose.staging.yml
version: '3.8'

services:
  app1:
    build:
      context: ./app1
      dockerfile: Dockerfile
      target: production
    environment:
      - DATABASE_URL=${STAGING_DATABASE_URL}
      - REDIS_URL=${STAGING_REDIS_URL}
      - DEBUG=false
    ports:
      - "8000:8000"
    restart: unless-stopped

  app2:
    build:
      context: ./app2
      dockerfile: Dockerfile
      target: production
    environment:
      - DATABASE_URL=${STAGING_DATABASE_URL}
      - REDIS_URL=${STAGING_REDIS_URL}
      - DEBUG=false
    ports:
      - "8001:8001"
    restart: unless-stopped

  web-ui:
    build:
      context: ./web-ui
      dockerfile: Dockerfile
      target: production
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=${STAGING_API_URL}
    ports:
      - "3000:3000"
    restart: unless-stopped
```

## Production Deployment

### 1. Production Preparation

```bash
# Create production branch
git checkout -b production

# Set up production secrets
mkdir -p secrets
./scripts/manage-secrets.sh generate-key > secrets/secret_key.txt
openssl rand -base64 32 > secrets/db_password.txt
openssl rand -base64 32 > secrets/redis_password.txt

# Set proper permissions
chmod 600 secrets/*.txt
```

### 2. Production Configuration

```bash
# Create production environment
cp env.production .env
# Edit .env with production values

# Validate configuration
make secrets-validate
```

### 3. Deploy with Secrets

```bash
# Deploy with Docker secrets
docker-compose -f docker-compose.secrets.yml up -d

# Check deployment
make health-check

# Monitor system
make monitor
```

### 4. Production Services

- **Web UI**: https://your-domain.com
- **App1 API**: https://api.your-domain.com
- **App2 API**: https://ea-api.your-domain.com
- **Monitoring**: https://monitoring.your-domain.com

## Docker Deployment

### 1. Build Images

```bash
# Build all images
make build

# Build specific service
docker-compose build app1
docker-compose build app2
docker-compose build web-ui
```

### 2. Deploy with Docker Compose

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d

# Development deployment
docker-compose -f docker-compose.dev.yml up -d

# Secrets deployment
docker-compose -f docker-compose.secrets.yml up -d
```

### 3. Docker Commands

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale app1=3

# Update services
docker-compose pull
docker-compose up -d
```

## Kubernetes Deployment

### 1. Create Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: axf-bot
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: axf-bot-config
  namespace: axf-bot
data:
  DATABASE_URL: "postgresql://postgres:password@postgres:5432/axf_bot_db"
  REDIS_URL: "redis://redis:6379/0"
  DEBUG: "false"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: axf-bot-secrets
  namespace: axf-bot
type: Opaque
data:
  secret-key: <base64-encoded-secret>
  db-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create configmap
kubectl apply -f k8s/configmap.yaml

# Create secrets
kubectl apply -f k8s/secret.yaml

# Deploy services
kubectl apply -f k8s/
```

### 3. Kubernetes Commands

```bash
# View pods
kubectl get pods -n axf-bot

# View services
kubectl get services -n axf-bot

# View logs
kubectl logs -f deployment/app1 -n axf-bot

# Scale deployment
kubectl scale deployment app1 --replicas=3 -n axf-bot
```

## Monitoring and Maintenance

### 1. Health Monitoring

```bash
# Check system health
make health-check

# Monitor system
make monitor

# Watch system status
make watch
```

### 2. Log Management

```bash
# View all logs
make dev-logs

# View specific service logs
docker-compose logs -f app1
docker-compose logs -f app2
docker-compose logs -f web-ui

# View database logs
docker-compose logs -f postgres
```

### 3. Database Maintenance

```bash
# Backup database
make db-backup

# Restore database
make db-restore backup_file.sql

# Run migrations
make migrate

# Seed database
make seed-db
```

### 4. System Updates

```bash
# Update code
git pull origin main

# Rebuild images
make build

# Deploy updates
docker-compose up -d

# Check deployment
make health-check
```

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
docker-compose exec postgres pg_isready -U postgres

# Check database logs
docker-compose logs postgres

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

# Check Docker logs
docker-compose logs
```

#### 4. Environment Issues

```bash
# Check environment status
make secrets-status

# Validate environment
make secrets-validate

# Recreate environment
rm .env
make secrets-create
```

### Debug Commands

```bash
# Check system status
make status

# View system resources
docker stats

# Check disk usage
df -h

# Check memory usage
free -h

# Check network connectivity
ping google.com
```

### Log Analysis

```bash
# Search for errors
docker-compose logs | grep -i error

# Search for warnings
docker-compose logs | grep -i warning

# Follow logs in real-time
docker-compose logs -f --tail=100
```

## Security Considerations

### 1. Environment Variables

- Never commit secrets to version control
- Use environment-specific configuration files
- Rotate secrets regularly
- Use Docker secrets in production

### 2. Network Security

- Use HTTPS in production
- Configure firewall rules
- Use VPN for database access
- Implement rate limiting

### 3. Database Security

- Use strong passwords
- Enable SSL connections
- Regular security updates
- Backup encryption

### 4. Container Security

- Use non-root users
- Scan images for vulnerabilities
- Regular security updates
- Resource limits

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_market_data_symbol ON market_data(symbol);

-- Analyze tables
ANALYZE strategies;
ANALYZE market_data;
```

### 2. Application Optimization

```bash
# Enable gzip compression
# Add to nginx configuration
gzip on;
gzip_types text/plain application/json;

# Enable caching
# Add to application headers
Cache-Control: max-age=3600
```

### 3. Container Optimization

```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as base
FROM base as production
# Only copy necessary files
COPY --from=base /app /app
```

## Backup and Recovery

### 1. Database Backup

```bash
# Create backup
pg_dump -h localhost -U postgres -d axf_bot_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql -h localhost -U postgres -d axf_bot_db < backup_file.sql
```

### 2. Application Backup

```bash
# Backup application data
tar -czf app_backup_$(date +%Y%m%d_%H%M%S).tar.gz app1/data app2/data

# Restore application data
tar -xzf app_backup_file.tar.gz
```

### 3. Configuration Backup

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz .env secrets/ deployment/

# Restore configuration
tar -xzf config_backup_file.tar.gz
```

## Scaling

### 1. Horizontal Scaling

```bash
# Scale services
docker-compose up -d --scale app1=3 --scale app2=2

# Use load balancer
# Configure nginx or HAProxy
```

### 2. Vertical Scaling

```bash
# Increase container resources
# Update docker-compose.yml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
```

### 3. Database Scaling

```bash
# Use read replicas
# Configure master-slave replication
# Use connection pooling
```

---

For more information, see the [Development Guide](DEVELOPMENT_GUIDE.md) and [Database Schema](DATABASE_SCHEMA.md) documentation.
