---
title: Embedding
description: Create text embeddings with any provider
---

The `embedding` and `aembedding` functions create vector embeddings from text using a unified interface across all providers that support embeddings.

## `any_llm.embedding()`

```python
def embedding(
    model: str,
    inputs: str | list[str],
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CreateEmbeddingResponse
```

## `any_llm.aembedding()`

Async variant with the same parameters.

```python
async def aembedding(
    model: str,
    inputs: str | list[str],
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CreateEmbeddingResponse
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. **Recommended**: use with a separate `provider` parameter (e.g., `model="text-embedding-3-small"`). **Alternative**: combined `"provider:model"` format. |
| `inputs` | `str \| list[str]` | *required* | The text to embed. Pass a single string or a list of strings for batch embedding. |
| `provider` | `str \| LLMProvider \| None` | `None` | Provider name (e.g., `"openai"`, `"voyage"`). |
| `api_key` | `str \| None` | `None` | API key. Falls back to the provider's environment variable. |
| `api_base` | `str \| None` | `None` | Override the provider's base URL. |
| `client_args` | `dict[str, Any] \| None` | `None` | Provider-specific arguments for client initialization. |
| `**kwargs` | `Any` | | Additional provider-specific arguments for the API call. |

## Return Value

Returns a [`CreateEmbeddingResponse`](/any-llm/api/types/completion/) containing:

- `data` -- list of `Embedding` objects, each with an `embedding` vector (`list[float]`) and an `index`.
- `model` -- the model used.
- `usage` -- token usage information with `prompt_tokens` and `total_tokens`.

## Usage

### Single text

```python
from any_llm import embedding

result = embedding(
    model="text-embedding-3-small",
    provider="openai",
    inputs="Hello, world!",
)

vector = result.data[0].embedding
print(f"Dimensions: {len(vector)}")
print(f"Tokens used: {result.usage.total_tokens}")
```

### Batch embedding

```python
result = embedding(
    model="text-embedding-3-small",
    provider="openai",
    inputs=["First sentence", "Second sentence", "Third sentence"],
)

for item in result.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```

### Async

```python
import asyncio
from any_llm import aembedding

async def main():
    result = await aembedding(
        model="text-embedding-3-small",
        provider="openai",
        inputs="Hello, world!",
    )
    print(f"Dimensions: {len(result.data[0].embedding)}")

asyncio.run(main())
```

:::note
Not all providers support embeddings. Check the [providers page](/any-llm/providers/) for support details, or query `ProviderMetadata.embedding` programmatically.
:::
