# Quick Start

## Run from Docker Image

First, generate a secure master key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

```bash
docker run \
-e GATEWAY_MASTER_KEY="your-secure-master-key" \
-e OPENAI_API_KEY="your-api-key" \
-p 8000:8000 \
ghcr.io/mozilla-ai/any-llm/gateway:latest
```

> **Note:** Instead of `latest`, you can use a specific release version (e.g., `1.2.0`). See [available versions](https://github.com/orgs/mozilla-ai/packages/container/package/any-llm%2Fgateway) on GitHub Packages.

## Local development

### Option 1: Docker compose

First, create a `config.yaml` file with your configuration. You can copy `docker/config.example.yaml` as a starting point:

```bash
cp docker/config.example.yaml docker/config.yaml
# Edit docker/config.yaml with your settings
```

Then run the Docker containers:

```bash
docker-compose -f docker/docker-compose.yml up -d --build

# Tail the logs
docker-compose -f docker/docker-compose.yml logs -f
```

This will run any-llm-gateway using the credentials and configuration specified in `config.yaml`.

### Option 2: CLI

In order for this to work, you will need to have a Postgres DB running.
```bash
uv venv --python=3.13
source .venv/bin/activate
uv sync --all-extras -U
```

```bash
export GATEWAY_MASTER_KEY="your-secure-master-key"
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/any_llm_gateway"
export OPENAI_API_KEY="your-api-key" # Or GEMINI_API_KEY etc

any-llm-gateway serve # Or, you can put the env vars in a config.yaml file and run serve with --config path/to/yaml
```

## Basic Usage

The gateway supports two authentication patterns for making completion requests:

### Option 1: Direct Master Key Authentication

First, create a user.

```bash
curl -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123", "alias": "Alice"}'
```

Use the master key directly and specify which user is making the request.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "user": "user-123"
  }'
```

### Option 2: Virtual API Keys

Create a virtual API key (you can optionally pass in a user_id too if you want the key linked to a user)

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{"key_name": "mobile-app"}'
```

Now you can use that new api key and don't need to pass in the user field.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer gw-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Usage is automatically tracked under the virtual user associated with the virtual key.
