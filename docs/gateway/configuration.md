# Configuration

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

### Dynamic Pricing via API

You can also set or update pricing dynamically using the API:
```bash
curl -X POST http://localhost:8000/v1/pricing \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4",
    "input_price_per_million": 30.0,
    "output_price_per_million": 60.0
  }'
```

This is useful for:  
- Updating pricing without restarting the gateway  
- Managing pricing in production environments  
- Adjusting rates as provider pricing changes  

**Important notes:**  
- Database pricing takes precedence - config only sets initial values  
- If pricing for the model already exists in the database, config values are ignored (with a warning logged)

## Next Steps

- See [supported providers](https://mozilla-ai.github.io/any-llm/providers/) for provider-specific configuration
- Learn about [authentication methods](./authentication.md) for managing access
- Set up [budget management](./budget-management.md) to enforce spending limits
