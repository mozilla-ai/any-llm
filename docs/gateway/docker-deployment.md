# Docker Deployment Guide

This guide walks you through deploying `any-llm-gateway` using Docker and Docker Compose. Whether you're setting up a local development environment or deploying to production, this guide covers the essential steps and best practices for a secure, reliable deployment.

## Quick Start with Docker Compose

Docker Compose is the recommended deployment method for most users. It automatically sets up both the gateway application and a PostgreSQL database with proper networking and dependencies.

**Prerequisites:**
- Docker Engine 20.10 or newer
- Docker Compose 2.0 or newer
- At least one LLM provider API key (OpenAI, Anthropic, Mistral, etc.)

### Configure the Gateway

First, prepare your configuration file with credentials and settings:

Copy the example configuration file:

```bash
cp docker/config.example.yml docker/config.yml
```

Generate a secure master key (minimum 32 characters recommended):

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save the output of this command for the next step. [Learn more about keys here](authentication.md).

Edit `docker/config.yml` with your master key and provider credentials. See the [Configuration Guide](configuration.md) for detailed options.

### Start the Services

Launch the gateway and database with a single command:

```bash
docker-compose -f docker/docker-compose.yml up -d
```

This command will:
- Pull the PostgreSQL 16 Alpine image
- Build the gateway Docker image from source (or pull from GHCR if configured)
- Create a dedicated network for service communication
- Start PostgreSQL with automatic health checks
- Wait for the database to be healthy before starting the gateway
- Initialize database tables and schema automatically

The `-d` flag runs services in detached mode (background).

### Verify Deployment

Confirm everything is running correctly:

```bash
# Test the health endpoint
curl http://localhost:8000/health
# Expected: {"status": "healthy"}

# Check service status
docker-compose -f docker/docker-compose.yml ps

# View real-time logs
docker-compose -f docker/docker-compose.yml logs -f gateway
```

If the health check returns successfully, your gateway is ready to accept requests!

## Standalone Docker Deployment

For scenarios where you have an existing PostgreSQL database or prefer more control over your deployment architecture, you can run the gateway as a standalone container.

### Using Pre-built Image

Pull and run the official image from GitHub Container Registry:

```bash
docker pull ghcr.io/mozilla-ai/any-llm/gateway:latest

docker run -d \
  --name any-llm-gateway \
  -p 8000:8000 \
  -v $(pwd)/config.yml:/app/config.yml \
  -e DATABASE_URL="postgresql://user:pass@host:5432/dbname" \
  ghcr.io/mozilla-ai/any-llm/gateway:latest \
  any-llm-gateway serve --config /app/config.yml
```

Replace the `DATABASE_URL` with your actual PostgreSQL connection string. The format is: `postgresql://username:password@hostname:port/database_name`

### Building from Source

If you need to customize the image or test local changes:

```bash
docker build -t any-llm-gateway:local -f docker/Dockerfile .

docker run -d \
  --name any-llm-gateway \
  -p 8000:8000 \
  -v $(pwd)/config.yml:/app/config.yml \
  -e DATABASE_URL="postgresql://user:pass@host:5432/dbname" \
  any-llm-gateway:local
```

## Production Deployment

Production deployments require additional considerations for reliability, security, and performance.

### Production Configuration

Enhance your docker-compose.yml with production-grade settings:

```yaml
services:
  gateway:
    image: ghcr.io/mozilla-ai/any-llm/gateway:latest
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Nginx Reverse Proxy

For production, always use a reverse proxy with HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name gateway.yourdomain.com;

    ssl_certificate /etc/ssl/certs/gateway.crt;
    ssl_certificate_key /etc/ssl/private/gateway.key;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for LLM streaming
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

## Environment Variables

The gateway can be configured using environment variables instead of or in addition to a config file. This is useful for Docker deployments and follows 12-factor app principles.

For a complete list of environment variables and configuration options, see the [Configuration Guide](configuration.md).

**Docker Compose example with .env file:**

```yaml
services:
  gateway:
    env_file:
      - .env
```

## Database Backups

```bash
# Backup
docker-compose -f docker/docker-compose.yml exec postgres \
  pg_dump -U gateway gateway > backup.sql

# Restore
docker-compose -f docker/docker-compose.yml exec -T postgres \
  psql -U gateway gateway < backup.sql
```

## Security Best Practices

1. **Never commit secrets** - Use `.env` files (gitignored) or Docker secrets
2. **Use read-only volumes** - Mount configs with `:ro` flag
3. **Enable HTTPS** - Use a reverse proxy with SSL certificates
4. **Isolate networks** - Keep database on internal network only
5. **Update regularly** - Use tagged versions and update containers periodically

## Monitoring and Logging

### Health Checks

```bash
# Test health endpoint
curl http://localhost:8000/health

# Check container health status
docker inspect --format='{{.State.Health.Status}}' container-name
```

### Logging

```bash
# View logs
docker-compose logs -f gateway

# Last 100 lines
docker-compose logs --tail=100 gateway
```

Configure log rotation:

```yaml
services:
  gateway:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Troubleshooting

**Container won't start:**
```bash
docker-compose logs gateway
```
Common issues: Database connection failed, port in use, missing config

**Database connection issues:**
```bash
docker-compose exec postgres psql -U gateway -c "SELECT version();"
```

**Permission errors:**
```bash
chmod 644 docker/config.yml
chmod 600 docker/service_account.json
```

**Rebuild after changes:**
```bash
docker-compose -f docker/docker-compose.yml up -d --build
```

## Next Steps

- [Configuration Guide](configuration.md) - Advanced configuration options
- [Authentication](authentication.md) - Set up API keys and user management
- [Budget Management](budget-management.md) - Configure spending limits
- [API Reference](api-reference.md) - Explore the complete API
- [Troubleshooting](troubleshooting.md) - Common issues and solutions
