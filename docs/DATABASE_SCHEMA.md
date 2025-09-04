# Database Schema Documentation

This document describes the database schema for the AXF Bot 0 project, including table structures, relationships, and usage patterns.

## Overview

The database uses PostgreSQL and consists of several main entities:
- **Strategies**: Trading strategies and their parameters
- **Strategy Performance**: Performance metrics and results
- **Expert Advisors**: Generated MT4 Expert Advisor files
- **Market Data**: Historical market data and indicators
- **News Data**: News articles and sentiment analysis
- **User Data**: User accounts and preferences

## Database Connection

### Connection String
```
postgresql://postgres:password@localhost:5432/axf_bot_db
```

### Environment Variables
```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/axf_bot_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=axf_bot_db
DB_USER=postgres
DB_PASSWORD=password
```

## Table Schema

### 1. Strategies Table

Stores trading strategies and their configuration parameters.

```sql
CREATE TABLE strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    strategy_type VARCHAR(100) NOT NULL,
    parameters JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100),
    version INTEGER DEFAULT 1
);
```

**Columns**:
- `id`: Primary key
- `name`: Strategy name
- `description`: Strategy description
- `strategy_type`: Type of strategy (e.g., 'trend_following', 'mean_reversion')
- `parameters`: JSONB object containing strategy parameters
- `status`: Strategy status ('active', 'inactive', 'testing')
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `created_by`: Creator identifier
- `version`: Strategy version number

**Indexes**:
```sql
CREATE INDEX idx_strategies_type ON strategies(strategy_type);
CREATE INDEX idx_strategies_status ON strategies(status);
CREATE INDEX idx_strategies_created_at ON strategies(created_at);
CREATE INDEX idx_strategies_parameters ON strategies USING GIN(parameters);
```

### 2. Strategy Performance Table

Tracks performance metrics for each strategy.

```sql
CREATE TABLE strategy_performance (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(50),
    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Columns**:
- `id`: Primary key
- `strategy_id`: Foreign key to strategies table
- `metric_name`: Name of the metric (e.g., 'total_return', 'sharpe_ratio')
- `metric_value`: Metric value
- `metric_unit`: Unit of measurement
- `period_start`: Start of measurement period
- `period_end`: End of measurement period
- `created_at`: Record creation timestamp

**Indexes**:
```sql
CREATE INDEX idx_strategy_performance_strategy_id ON strategy_performance(strategy_id);
CREATE INDEX idx_strategy_performance_metric_name ON strategy_performance(metric_name);
CREATE INDEX idx_strategy_performance_period ON strategy_performance(period_start, period_end);
```

### 3. Expert Advisors Table

Stores generated MetaTrader 4 Expert Advisor files.

```sql
CREATE TABLE expert_advisors (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_content TEXT NOT NULL,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'generated',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Columns**:
- `id`: Primary key
- `strategy_id`: Foreign key to strategies table
- `name`: EA name
- `file_path`: Path to the EA file
- `file_content`: MQL4 code content
- `version`: EA version
- `status`: EA status ('generated', 'tested', 'deployed')
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp

**Indexes**:
```sql
CREATE INDEX idx_expert_advisors_strategy_id ON expert_advisors(strategy_id);
CREATE INDEX idx_expert_advisors_status ON expert_advisors(status);
CREATE INDEX idx_expert_advisors_created_at ON expert_advisors(created_at);
```

### 4. Market Data Table

Stores historical market data and technical indicators.

```sql
CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(15,6) NOT NULL,
    high_price DECIMAL(15,6) NOT NULL,
    low_price DECIMAL(15,6) NOT NULL,
    close_price DECIMAL(15,6) NOT NULL,
    volume BIGINT,
    indicators JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Columns**:
- `id`: Primary key
- `symbol`: Currency pair symbol (e.g., 'EURUSD')
- `timeframe`: Timeframe (e.g., '1H', '4H', '1D')
- `timestamp`: Data timestamp
- `open_price`: Opening price
- `high_price`: Highest price
- `low_price`: Lowest price
- `close_price`: Closing price
- `volume`: Trading volume
- `indicators`: JSONB object containing technical indicators
- `created_at`: Record creation timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_market_data_unique ON market_data(symbol, timeframe, timestamp);
CREATE INDEX idx_market_data_symbol ON market_data(symbol);
CREATE INDEX idx_market_data_timeframe ON market_data(timeframe);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_market_data_indicators ON market_data USING GIN(indicators);
```

### 5. News Data Table

Stores news articles and sentiment analysis results.

```sql
CREATE TABLE news_data (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    url VARCHAR(500),
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    sentiment_score DECIMAL(3,2),
    sentiment_label VARCHAR(20),
    relevance_score DECIMAL(3,2),
    keywords TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

**Columns**:
- `id`: Primary key
- `title`: News article title
- `content`: Article content
- `source`: News source
- `url`: Article URL
- `published_at`: Publication timestamp
- `sentiment_score`: Sentiment score (-1 to 1)
- `sentiment_label`: Sentiment label ('positive', 'negative', 'neutral')
- `relevance_score`: Relevance score (0 to 1)
- `keywords`: Array of relevant keywords
- `created_at`: Record creation timestamp

**Indexes**:
```sql
CREATE INDEX idx_news_data_published_at ON news_data(published_at);
CREATE INDEX idx_news_data_sentiment_score ON news_data(sentiment_score);
CREATE INDEX idx_news_data_sentiment_label ON news_data(sentiment_label);
CREATE INDEX idx_news_data_relevance_score ON news_data(relevance_score);
CREATE INDEX idx_news_data_keywords ON news_data USING GIN(keywords);
```

### 6. Users Table

Stores user accounts and preferences.

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);
```

**Columns**:
- `id`: Primary key
- `username`: Unique username
- `email`: Unique email address
- `password_hash`: Hashed password
- `first_name`: User's first name
- `last_name`: User's last name
- `is_active`: Account active status
- `is_admin`: Admin privileges
- `preferences`: User preferences as JSONB
- `created_at`: Account creation timestamp
- `updated_at`: Last update timestamp
- `last_login`: Last login timestamp

**Indexes**:
```sql
CREATE UNIQUE INDEX idx_users_username ON users(username);
CREATE UNIQUE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_active ON users(is_active);
CREATE INDEX idx_users_is_admin ON users(is_admin);
```

## Relationships

### Foreign Key Relationships

1. **strategy_performance.strategy_id** → **strategies.id**
   - One strategy can have many performance records
   - Cascade delete when strategy is deleted

2. **expert_advisors.strategy_id** → **strategies.id**
   - One strategy can have many Expert Advisors
   - Cascade delete when strategy is deleted

### Data Flow

```
Market Data → Strategy Generation → Strategies → Performance Tracking
     ↓              ↓                    ↓              ↓
News Data → Sentiment Analysis → Expert Advisors → Monitoring
```

## Common Queries

### 1. Get Strategy Performance

```sql
SELECT 
    s.name,
    s.strategy_type,
    sp.metric_name,
    sp.metric_value,
    sp.period_start,
    sp.period_end
FROM strategies s
JOIN strategy_performance sp ON s.id = sp.strategy_id
WHERE s.status = 'active'
ORDER BY sp.period_end DESC;
```

### 2. Get Top Performing Strategies

```sql
SELECT 
    s.name,
    s.strategy_type,
    AVG(sp.metric_value) as avg_performance
FROM strategies s
JOIN strategy_performance sp ON s.id = sp.strategy_id
WHERE sp.metric_name = 'total_return'
    AND sp.period_end >= NOW() - INTERVAL '30 days'
GROUP BY s.id, s.name, s.strategy_type
ORDER BY avg_performance DESC
LIMIT 10;
```

### 3. Get Recent Market Data

```sql
SELECT 
    symbol,
    timeframe,
    timestamp,
    close_price,
    indicators->>'rsi' as rsi,
    indicators->>'macd' as macd
FROM market_data
WHERE symbol = 'EURUSD'
    AND timeframe = '1H'
    AND timestamp >= NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;
```

### 4. Get News Sentiment

```sql
SELECT 
    title,
    sentiment_score,
    sentiment_label,
    published_at
FROM news_data
WHERE sentiment_score IS NOT NULL
    AND published_at >= NOW() - INTERVAL '7 days'
ORDER BY published_at DESC;
```

## Database Maintenance

### Backup

```bash
# Create backup
pg_dump -h localhost -U postgres -d axf_bot_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql -h localhost -U postgres -d axf_bot_db < backup_file.sql
```

### Vacuum and Analyze

```sql
-- Vacuum tables
VACUUM ANALYZE strategies;
VACUUM ANALYZE strategy_performance;
VACUUM ANALYZE market_data;
VACUUM ANALYZE news_data;

-- Reindex
REINDEX DATABASE axf_bot_db;
```

### Monitoring

```sql
-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## Migration Scripts

### Create Migration

```bash
# Create new migration
python scripts/migrate.py --create "add_new_table"

# Apply migrations
python scripts/migrate.py --migrate

# Rollback migration
python scripts/migrate.py --rollback
```

### Sample Migration

```python
# migrations/001_add_new_table.py
def up(connection):
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE new_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
    """)
    connection.commit()

def down(connection):
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS new_table;")
    connection.commit()
```

## Performance Optimization

### Indexing Strategy

1. **Primary Keys**: Automatically indexed
2. **Foreign Keys**: Indexed for join performance
3. **Frequently Queried Columns**: Indexed based on query patterns
4. **JSONB Columns**: GIN indexes for efficient JSON queries
5. **Composite Indexes**: For multi-column queries

### Query Optimization

1. **Use EXPLAIN ANALYZE**: To understand query execution plans
2. **Avoid SELECT ***: Select only needed columns
3. **Use LIMIT**: For large result sets
4. **Proper WHERE clauses**: Use indexed columns
5. **Connection pooling**: Use connection pooling for high concurrency

### Monitoring

```sql
-- Check slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Check table bloat
SELECT 
    schemaname,
    tablename,
    n_dead_tup,
    n_live_tup,
    n_dead_tup::float / n_live_tup::float as dead_ratio
FROM pg_stat_user_tables
WHERE n_live_tup > 0
ORDER BY dead_ratio DESC;
```

## Security Considerations

### Access Control

```sql
-- Create read-only user
CREATE USER readonly_user WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE axf_bot_db TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- Create application user
CREATE USER app_user WITH PASSWORD 'app_password';
GRANT CONNECT ON DATABASE axf_bot_db TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
```

### Data Encryption

1. **Connection Encryption**: Use SSL/TLS for database connections
2. **Column Encryption**: Encrypt sensitive data at rest
3. **Password Hashing**: Use bcrypt for password hashing
4. **API Keys**: Store API keys encrypted

### Audit Logging

```sql
-- Enable audit logging
CREATE EXTENSION IF NOT EXISTS pgaudit;

-- Log all DDL statements
ALTER SYSTEM SET pgaudit.log = 'ddl';
SELECT pg_reload_conf();
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if PostgreSQL is running
2. **Authentication Failed**: Verify username/password
3. **Database Does Not Exist**: Create database first
4. **Permission Denied**: Check user privileges
5. **Lock Timeout**: Check for long-running transactions

### Debug Commands

```bash
# Check PostgreSQL status
systemctl status postgresql

# Check database connections
psql -h localhost -U postgres -c "SELECT * FROM pg_stat_activity;"

# Check database size
psql -h localhost -U postgres -c "SELECT pg_size_pretty(pg_database_size('axf_bot_db'));"

# Check table locks
psql -h localhost -U postgres -c "SELECT * FROM pg_locks WHERE NOT granted;"
```

---

For more information, see the [Development Guide](DEVELOPMENT_GUIDE.md) and [Environment Setup](ENVIRONMENT_SETUP.md) documentation.
