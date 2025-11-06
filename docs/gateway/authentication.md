# Authentication

The any-llm Gateway supports two main authentication patterns for making completion requests.

## Direct Master Key Authentication

Use the master key directly and specify which user is making the request.

### Creating a User

```bash
curl -X POST http://localhost:8000/v1/users \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user-123", "alias": "Alice"}'
```

### Making Requests with Master Key

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

## Virtual API Keys

Virtual API keys provide a more secure way to authenticate requests without exposing the master key.

### Creating a Virtual API Key

```bash
curl -X POST http://localhost:8000/v1/keys \
  -H "X-AnyLLM-Key: Bearer your-secure-master-key" \
  -H "Content-Type: application/json" \
  -d '{"key_name": "mobile-app"}'
```

Response:
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

### Using Virtual API Keys

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer gw-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Usage is automatically tracked under the virtual user associated with the virtual key.
