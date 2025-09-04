# Secrets Management

This directory contains Docker secrets for production deployment.

## Security Notice

⚠️ **IMPORTANT**: Never commit actual secrets to version control!

## Setup Instructions

1. **Create the secrets directory** (if not exists):
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

4. **Add to .gitignore**:
   ```bash
   echo "secrets/*.txt" >> .gitignore
   ```

## File Structure

```
secrets/
├── README.md              # This file
├── secret_key.txt         # Application secret key
├── db_password.txt        # Database password
├── redis_password.txt     # Redis password
├── influxdb_password.txt  # InfluxDB password
├── fano_api_key.txt       # Fano API key
└── news_api_key.txt       # News API key
```

## Usage

### Development
Use the regular `docker-compose.yml` or `docker-compose.dev.yml` with environment variables.

### Production
Use `docker-compose.secrets.yml` with Docker secrets:

```bash
# Start with secrets
docker-compose -f docker-compose.secrets.yml up -d

# Check secrets
docker-compose -f docker-compose.secrets.yml exec app1 cat /run/secrets/secret_key
```

## Environment Variables

The following environment variables are used:

- `SECRET_KEY_FILE` - Path to secret key file
- `DB_PASSWORD_FILE` - Path to database password file
- `REDIS_PASSWORD_FILE` - Path to Redis password file
- `INFLUXDB_PASSWORD_FILE` - Path to InfluxDB password file
- `FANO_API_KEY_FILE` - Path to Fano API key file
- `NEWS_API_KEY_FILE` - Path to News API key file

## Security Best Practices

1. **Never commit secrets** to version control
2. **Use strong passwords** (32+ characters)
3. **Rotate secrets regularly**
4. **Use different secrets** for different environments
5. **Restrict file permissions** (600)
6. **Use Docker secrets** in production
7. **Monitor access** to secret files
8. **Use environment-specific** secret files

## Troubleshooting

### Permission Denied
```bash
chmod 600 secrets/*.txt
```

### Secret Not Found
```bash
# Check if secret file exists
ls -la secrets/

# Check Docker secret
docker secret ls
```

### Invalid Secret Format
```bash
# Check secret content (first few characters)
head -c 10 secrets/secret_key.txt
```
