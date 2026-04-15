---
title: Embedding
description: Create text embeddings with any provider
---

The `embedding` and `aembedding` functions create vector embeddings from text using a unified interface across all providers that support embeddings.

## `any_llm.embedding()`

```
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

```
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
| `model` | `str` | *required* | Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='gpt-4', provider='openai'). **Alternative**: Combined format 'provider:model' (e.g., 'openai:gpt-4'). Legacy format 'provider/model' is also supported but deprecated. |
| `inputs` | `str \| list[str]` | *required* | The input text to embed |
| `provider` | `str \| LLMProvider \| None` | None | **Recommended**: Provider name to use for the request (e.g., 'openai', 'mistral'). When provided, the model parameter should contain only the model name. |
| `api_key` | `str \| None` | None | API key for the provider |
| `api_base` | `str \| None` | None | Base URL for the provider API |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments that will be passed to the provider's client instantiation. |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments that will be passed to the provider's API call. |

## Return Value

Returns a [`CreateEmbeddingResponse`](types/completion.md) containing:

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

{% hint style="info" %}
Not all providers support embeddings. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.embedding` programmatically.
{% endhint %}
