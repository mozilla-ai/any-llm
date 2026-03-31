---
title: Quick Start
description: Set up any-llm-gateway and make your first LLM completion request
---

This guide will help you set up any-llm-gateway and make your first LLM completion request. The gateway acts as a proxy between your applications and LLM providers, providing cost control, usage tracking, and API key management.

By the end of this guide, you will:

1. Configure provider credentials and model pricing (e.g., OpenAI API key)
1. Run the gateway
1. Authenticate requests using a bootstrap gateway key
1. Make completion requests through the gateway

> **Note:** This quickstart uses `uvx` with SQLite as the default local database. Docker + Postgres is included below as an optional production-style setup.

## Pre-Requisites

1. `uv` installed
1. Access to at least one LLM provider

## Configure and run the Gateway

When running any-llm-gateway, it must have a few things configured:

1. `GATEWAY_MASTER_KEY`. This master key has admin access to manage budgets, users, virtual keys, etc.
1. `DATABASE_URL` (optional). By default, gateway uses `sqlite:///./any-llm-gateway.db`.
1. Provider Keys. The gateway connects to providers (Mistral, AWS, Vertex, Azure, etc) using credentials that must be set.

### Create a project directory
```bash
mkdir any-llm-gateway
cd any-llm-gateway
```

### Generate master key

First, generate a secure master key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Save the output of this command, you'll need it in the next steps.

### Configure providers

Create a file name `config.yml` and paste the below content:

> **Action** : At a minimum you'll need to fill out the master_key, and also enter credential information for at least one provider. You can browse supported providers [here](https://mozilla-ai.github.io/any-llm/providers/). If you would like to track usage cost, you'll also need to configure model pricing, as explained in the [config template file](https://raw.githubusercontent.com/mozilla-ai/any-llm/main/docker/config.example.yml).

```yaml
database_url: "postgresql://gateway:gateway@postgres:5432/gateway"

master_key: 09kS0xTiz6JqO....

providers:
  openai:
    api_key: YOUR_OPENAI_API_KEY_HERE
    api_base: "https://api.openai.com/v1"  # optional, useful when you want to use a specific version of the API

models:
  openai:gpt-4:
    input_price_per_million: 0.15
    output_price_per_million: 0.60
```

### Start the gateway with `uvx`

Set a provider key (example with OpenAI):

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

Start the gateway:

```bash
uvx --from 'any-llm-sdk[gateway]' gateway serve --config config.yml
```

> **Deprecated command:** `any-llm-gateway serve --config config.yml` still works, but will be removed in a future major version.

Verify the gateway is running:

```bash
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}
```

### Optional: Set up Docker Configuration

Create a file named `docker-compose.yml` with the following content.

<details>
<summary>Click to view docker-compose.yml content</summary>

```yaml
services:
  gateway:
    # Use the official production image
    image: ghcr.io/mozilla-ai/any-llm/gateway:latest
    ports:
      - "8000:8000"
    volumes:
      - ./config.yml:/app/config.yml
      # UNCOMMENT the next line ONLY if using Google Vertex AI (requires service_account.json)
      # - ./service_account.json:/app/service_account.json
    command: ["gateway", "serve", "--config", "/app/config.yml"]
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      - POSTGRES_USER=gateway
      - POSTGRES_PASSWORD=gateway
      - POSTGRES_DB=gateway
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gateway"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
```
</details>

**Alternatively**, you can download the file directly from the repository:

```bash
curl -o docker-compose.yml https://raw.githubusercontent.com/mozilla-ai/any-llm/main/docker/docker-compose.yml
```

### Start the gateway with Docker

```bash
# From project root directory
docker compose up -d
```

```bash
# Verify the gateway is running
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy"}
```

### View Logs

```bash
docker compose logs -f
```


## Make your first request (no user setup)

On first startup with an empty database, the gateway creates one bootstrap API key and prints it in logs.

Copy that key and set it in your terminal:

```bash
export GATEWAY_API_KEY=YOUR_BOOTSTRAP_GATEWAY_KEY
```

Now call the gateway using the any-llm SDK:

```python
from any_llm import completion

response = completion(
    provider="gateway",
    model="openai:gpt-4o",
    api_base="http://0.0.0.0:8000/v1",
    api_key="YOUR_BOOTSTRAP_GATEWAY_KEY",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

You can also test with curl:

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Admin endpoints still use master key

For management operations (`/v1/users`, `/v1/keys`, `/v1/budgets`), keep using the master key:

```bash
export GATEWAY_MASTER_KEY=YOUR_MASTER_KEY
```

## Next Steps

- **[Configuration](/any-llm/gateway/configuration/)** - Configure providers, pricing, and other settings
- **[Authentication](/any-llm/gateway/authentication/)** - Learn about master keys and virtual API keys
- **[Budget Management](/any-llm/gateway/budget-management/)** - Set spending limits and track costs
- **[API Reference](/any-llm/gateway/api-reference/)** - Explore the complete API
