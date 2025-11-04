# Quick Start

## Run from Docker Image

> [!IMPORTANT]
> You need to [authenticate to the container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-with-a-personal-access-token-classic).

```bash
docker run \
-e GATEWAY_MASTER_KEY="your-secure-master-key" \
-e OPENAI_API_KEY="your-api-key" \
-p 8000:8000 \
ghcr.io/mozilla-ai/any-llm-gateway:latest
```


## Local development

> [!IMPORTANT]
> The following assumes that you have cloned this repository and `cd` into its root.

### Option 1: Docker compose

First, create a `config.yaml` file with your configuration, using config.example.yaml as a template.

Then run the Docker containers:

```bash
docker-compose up -d --build

# Tail the logs
docker-compose logs -f
```

This will run any-llm-gateway using the credentials and configuration specified in `config.yaml`.

### Option 2: CLI

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

Now you can use that new api key and don't need to pass in the user field.

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "X-AnyLLM-Key: Bearer gw-..." \
  -H "Content-Type: application/json" \
  -d '{"model": "openai:gpt-5-mini", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Usage is automatically tracked under the virtual user associated with the virtual key.

Response:

```json
{
    "id": "chatcmpl-CT9lmQ9yXkO5RUEbPYIAQFi5jECGl",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "logprobs": null,
            "message": {
                "content": "Hello! How can I assist you today?",
                "refusal": null,
                "role": "assistant",
                "annotations": [],
                "audio": null,
                "function_call": null,
                "tool_calls": null,
                "reasoning": null
            }
        }
    ],
    "created": 1761065102,
    "model": "gpt-4-0613",
    "object": "chat.completion",
    "service_tier": "default",
    "system_fingerprint": null,
    "usage": {
        "completion_tokens": 9,
        "prompt_tokens": 9,
        "total_tokens": 18,
        "completion_tokens_details": {
            "accepted_prediction_tokens": 0,
            "audio_tokens": 0,
            "reasoning_tokens": 0,
            "rejected_prediction_tokens": 0
        },
        "prompt_tokens_details": {
            "audio_tokens": 0,
            "cached_tokens": 0
        }
    }
}
```
