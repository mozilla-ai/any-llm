# Configuration

## Option 1: Config File

Create a `config.yml`:

> **Generating a secure master key:**
> `python -c "import secrets; print(secrets.token_urlsafe(32))"`

```yaml
database_url: "postgresql://gateway:gateway@localhost:5432/gateway_db"
master_key: "your-secure-master-key"

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
  gemini:
    api_key: "${GEMINI_API_KEY}"
  vertexai:
    credentials: "/path/to/service_account.json"
    project: "your-gcp-project-id"
    location: "us-central1"

pricing:
  openai:gpt-3.5-turbo:
    input_price_per_million: 0.5
    output_price_per_million: 1.5
```

Start with config file:
```bash
any-llm-gateway serve --config config.yml
```

## Option 2: Environment Variables

```bash
export DATABASE_URL="postgresql://gateway:gateway@localhost:5432/gateway_db"
export GATEWAY_MASTER_KEY="your-secure-master-key"
export GATEWAY_HOST="0.0.0.0"
export GATEWAY_PORT=8000

any-llm-gateway serve
```

## Model Pricing Configuration

Configure model pricing in your config file to automatically track costs. Pricing can be set via config file or dynamically via the API.

### Config File Pricing

Add pricing for models in your config file using the format `provider:model`:

```yaml
pricing:
  openai:gpt-3.5-turbo:
    input_price_per_million: 0.5
    output_price_per_million: 1.5
```

**Important notes:**
- Database pricing takes precedence - config only sets initial values
- If pricing for the model already exists in the database, config values are ignored (with a warning logged)
