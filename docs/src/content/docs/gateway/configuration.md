---
title: Configuration
description: Configure the gateway using YAML files or environment variables
---

The any-llm-gateway requires configuration to connect to your database, authenticate requests, and route to LLM providers. This guide covers the two main configuration approaches and how to set up model pricing for cost tracking.

You can configure the gateway using either a YAML configuration file or environment variables:

- **Config File (Recommended)**: Best for development and when managing multiple providers with complex settings. Easier to version control and share across teams.
- **Environment Variables**: Best for production deployments, containerized environments, or when following 12-factor app principles.

Both methods can also be combined—environment variables will override config file values.

## Option 1: Config File

Create a `config.yml` file with your database connection, master key, and provider credentials:

> **Generating a secure master key:**
> ```bash
>  python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```

```yaml
#Database connection
database_url: "postgresql://gateway:gateway@localhost:5432/gateway_db"

#Master key for admin access
master_key: "your-secure-master-key"

## LLM Provider Credentials
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
  gemini:
    api_key: "${GEMINI_API_KEY}"
  vertexai:
    credentials: "/path/to/service_account.json"
    project: "your-gcp-project-id"
    location: "us-central1"

# Model pricing for cost-tracking (optional)
pricing:
  openai:gpt-4:
    input_price_per_million: 0.15
    output_price_per_million: 0.6
    effective_at: "2025-05-01T00:00:00Z"  # optional, defaults to now
```

Start the gateway with your config file:

```bash
any-llm-gateway serve --config config.yml
```

## Option 2: Environment Variables
Configure the gateway entirely through environment variables—useful for containerized deployments:

```bash
#Required settings
export DATABASE_URL="postgresql://gateway:gateway@localhost:5432/gateway_db"
export GATEWAY_MASTER_KEY="your-secure-master-key"
export GATEWAY_HOST="0.0.0.0"
export GATEWAY_PORT=8000

any-llm-gateway serve
```
> **Note**: Model pricing cannot be set via environment variables. Use the config file or the [Pricing API](#dynamic-pricing-via-api) instead.


## Model Pricing Configuration

Configure model pricing in your config file to automatically track costs. Pricing can be set via config file or dynamically via the API.

### Config File Pricing

Add pricing for models in your config file using the format `provider:model-name`:

```yaml
pricing:
  openai:gpt-3.5-turbo:
    input_price_per_million: 0.5
    output_price_per_million: 1.5
```

Each pricing entry supports an optional `effective_at` field (ISO 8601 datetime) that records when this price takes effect. This enables accurate historical cost tracking when providers change their rates:

```yaml
pricing:
  openai:gpt-4:
    input_price_per_million: 30.0
    output_price_per_million: 60.0
    effective_at: "2025-01-01T00:00:00Z"
  openai:gpt-4:
    input_price_per_million: 25.0
    output_price_per_million: 50.0
    effective_at: "2025-06-01T00:00:00Z"
```

When `effective_at` is omitted, it defaults to the current time. Cost lookups always use the price that was in effect at the time of the API request.

### Dynamic Pricing via API

You can also set or update pricing dynamically using the API:
```bash
curl -X POST http://localhost:8000/v1/pricing \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model_key": "openai:gpt-4",
    "input_price_per_million": 30.0,
    "output_price_per_million": 60.0,
    "effective_at": "2025-06-01T00:00:00Z"
  }'
```

To view the full pricing history for a model:
```bash
curl http://localhost:8000/v1/pricing/openai:gpt-4/history
```

This is useful for:
- Updating pricing without restarting the gateway
- Managing pricing in production environments
- Recording price changes with specific effective dates
- Querying historical prices for auditing

**Important notes:**
- Pricing is keyed by `(model_key, effective_at)`. Multiple entries per model are supported for different effective dates.
- Database entries take precedence: if a config entry with the same model key and effective date already exists in the database, the config value is skipped.
- Cost is always computed using the price in effect at the time of the API request, not the latest price.

## Provider Client Args

You can set additional arguments to provider clients via the `client_args` configuration. These arguments are passed directly to the provider's client initialization, enabling custom headers, timeouts, and other provider-specific options.

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    client_args:
      custom_headers:
        X-Custom-Header: "custom-value"
      timeout: 60
```

Common use cases:
- **Custom headers**: Pass additional headers to the provider (e.g., for proxy authentication or request tracing)
- **Timeouts**: Configure connection and request timeouts
- **Provider-specific options**: Pass any additional arguments supported by the provider's client

The available `client_args` options depend on the provider. See the [any-llm provider documentation](https://mozilla-ai.github.io/any-llm/providers/) for provider-specific options.

## Next Steps

- See [supported providers](https://mozilla-ai.github.io/any-llm/providers/) for provider-specific configuration
- Learn about [authentication methods](/any-llm/gateway/authentication/) for managing access
- Set up [budget management](/any-llm/gateway/budget-management/) to enforce spending limits
