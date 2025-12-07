# Authentication

any-llm-gateway offers two authentication methods, each designed for different use cases. Understanding when to use each approach will help you secure your gateway effectively.
## Authentication Methods Overview

| Method | Best For | Key Management | Usage Tracking |
|--------|----------|----------------|----------------|
| **Master Key** | Internal services, admin operations, trusted environments | Single key with full access | Requires manual user specification |
| **Virtual API Keys** | External apps, per-user access, customer-facing services | Multiple scoped keys | Automatic per-key tracking |

### Supported Headers

The gateway accepts authentication via two headers:

- **`X-AnyLLM-Key`** (preferred): The gateway's native authentication header
- **`Authorization`**: Standard HTTP authorization header for OpenAI client compatibility

Both headers use the `Bearer <token>` format. When both headers are present, `X-AnyLLM-Key` takes precedence.

Using the `Authorization` header allows you to use the gateway with OpenAI-compatible clients without modification:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-master-key-or-virtual-key",  # Sent as Authorization: Bearer ...
)
```

## Master Key 
The master key is the root credential for your gateway installation. It has unrestricted access to all gateway operations and should be treated with the same security as your production database credentials.

### Generating a Master Key

Generate a cryptographically secure master key (minimum 32 characters recommended):

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Example output:**
```
Zx8Q_vKm3nR7wP2sT9yU5iO1eA6hD4fG0bN8cL3jM5k
```

Set the generated key in your configuration:

**Using environment variables:**
```bash
export GATEWAY_MASTER_KEY="Zx8Q_vKm3nR7wP2sT9yU5iO1eA6hD4fG0bN8cL3jM5k"
```

**Using config.yml:**
```yaml
master_key: "Zx8Q_vKm3nR7wP2sT9yU5iO1eA6hD4fG0bN8cL3jM5k"
```

### Creating a User

```bash
curl -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123", "alias": "Alice"}'
```
<details>
<summary>With optional metadata</summary>

```bash
curl -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d ' { 
    "user_id": "user-123", 
    "alias": "Alice",
    "metadata": {
      "department": "Engineering",
      "team": "ML",
      "email": "alice@example.com"
    }
  }'
```
</details>

### Making Requests with Master Key
When using the master key, you **must** specify which user is making the request using the `user` field:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai:gpt-4o-mini",
    "messages": [{"role": "user", "content": "Write a haiku on Jupiter"}],
    "user": "user-123"
  }'
```
The `user` field tells the gateway which user's budget and spend tracking to update. Without this field, the request will be rejected.

## Virtual API Keys

Virtual API keys provide scoped access for making completion requests without exposing the master key. Each virtual key can have expiration dates, metadata, and associated users for automatic usage tracking.

### Creating a Virtual API Key
Create a virtual key with a descriptive name : 

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"key_name": "mobile-app"}'
```
> **Important:** Save the `key` value immediately—it's only shown once and cannot be retrieved later.

<details>
<summary>Example Response</summary>

```json
{
  "id": "abc-123",
  "key": "gw-...",
  "key_name": "mobile-app",
  "created_at": "2025-10-20T10:00:00",
  "expires_at": null,
  "is_active": true,
  "metadata": {}
}
```
</details>

#### Key with Expiration

Create a key that automatically expires on a specific date:

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "key_name": "trial-access",
    "expires_at": "2025-12-31T23:59:59Z"
  }'
```

### Using Virtual API Keys
Making requests with a virtual key is simpler than using the master key—no `user` field is required:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer gw-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Write a haiku on Saturn"}]}'
```

The gateway automatically tracks usage based on the virtual key used.
### Managing Virtual Keys

#### List All Keys
**List all keys:**
```bash
curl http://localhost:8000/v1/keys \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}"
```

**Deactivate a key:**
```bash
curl -X PATCH http://localhost:8000/v1/keys/<virtual_key_id>\
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

**Delete a key:**
```bash
curl -X DELETE http://localhost:8000/v1/keys/<virtual_key_id> \
  -H "X-AnyLLM-Key: Bearer ${GATEWAY_MASTER_KEY}"
```

> See [API Reference](api-reference.md) for complete key management operations.


Note: The actual key values are never returned in list or get operations for security reasons.

## Next Steps

Now that you understand authentication, explore these related topics:

- **[Budget Management](budget-management.md)** - Set spending limits for users and enforce budgets
- **[Configuration](configuration.md)** - Learn about provider setup and pricing configuration
- **[API Reference](api-reference.md)** - Explore all available endpoints for managing keys and users
- **[Quick Start](quickstart.md)** - Complete walkthrough of setting up your first gateway

For questions or issues, refer to the [troubleshooting guide](troubleshooting.md) or check the project's issue tracker.
