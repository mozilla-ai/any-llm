#!/usr/bin/env bash
#
# Liveness check for the gateway Docker image.
# Starts PostgreSQL, runs the gateway container, and validates health endpoints.
#
# Usage: ./scripts/docker_liveness_check.sh <image_tag>
# Example: ./scripts/docker_liveness_check.sh gateway-test:abc123
#

set -euo pipefail

IMAGE_TAG="${1:-}"

if [[ -z "$IMAGE_TAG" ]]; then
    echo "Usage: $0 <image_tag>"
    echo "Example: $0 gateway-test:latest"
    exit 1
fi

POSTGRES_CONTAINER="test-postgres"
GATEWAY_CONTAINER="test-gateway"
CONFIG_FILE="/tmp/gateway-config.yml"

cleanup() {
    echo "Cleaning up containers..."
    docker rm -f "$GATEWAY_CONTAINER" "$POSTGRES_CONTAINER" 2>/dev/null || true
}

trap cleanup EXIT

echo "Starting PostgreSQL container..."
docker run -d --name "$POSTGRES_CONTAINER" \
    -e POSTGRES_USER=gateway \
    -e POSTGRES_PASSWORD=gateway \
    -e POSTGRES_DB=gateway \
    -p 5432:5432 \
    postgres:16-alpine

echo "Waiting for PostgreSQL to be ready..."
for i in {1..30}; do
    if docker exec "$POSTGRES_CONTAINER" pg_isready -U gateway > /dev/null 2>&1; then
        echo "PostgreSQL is ready"
        break
    fi
    if [[ $i -eq 30 ]]; then
        echo "PostgreSQL failed to become ready"
        docker logs "$POSTGRES_CONTAINER"
        exit 1
    fi
    sleep 1
done

echo "Creating gateway config..."
cat > "$CONFIG_FILE" << 'EOF'
database_url: "postgresql://gateway:gateway@host.docker.internal:5432/gateway"
host: "0.0.0.0"
port: 8000
master_key: "test-master-key"
EOF

echo "Starting gateway container with image: $IMAGE_TAG"
docker run -d --name "$GATEWAY_CONTAINER" \
    --add-host=host.docker.internal:host-gateway \
    -p 8000:8000 \
    -v "$CONFIG_FILE":/app/config.yml \
    "$IMAGE_TAG" \
    any-llm-gateway serve --config /app/config.yml

echo "Waiting for gateway to be healthy..."
for i in {1..60}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Gateway is responding"
        break
    fi
    if [[ $i -eq 60 ]]; then
        echo "Gateway failed to become healthy within 60 seconds"
        echo "=== Gateway logs ==="
        docker logs "$GATEWAY_CONTAINER"
        exit 1
    fi
    sleep 1
done

echo "Testing /health endpoint..."
curl -sf http://localhost:8000/health
echo ""

echo "Testing /health/liveness endpoint..."
curl -sf http://localhost:8000/health/liveness
echo ""

echo "Testing /health/readiness endpoint..."
curl -sf http://localhost:8000/health/readiness
echo ""

echo "All liveness checks passed!"
