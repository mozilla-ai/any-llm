---
title: List Models
description: List available models for a provider
---

The `list_models` and `alist_models` functions return the available models for a given provider.

## `any_llm.list_models()`

```python
def list_models(
    provider: str | LLMProvider,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Model]
```

## `any_llm.alist_models()`

Async variant with the same parameters.

```python
async def alist_models(
    provider: str | LLMProvider,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Model]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str \| LLMProvider` | *required* |  |
| `api_key` | `str \| None` | None |  |
| `api_base` | `str \| None` | None |  |
| `client_args` | `dict[str, Any] \| None` | None |  |
| `**kwargs` | `Any` | *required* |  |

## Return Value

Returns a `Sequence` of [`Model`](types/model.md) objects. Each `Model` has at minimum an `id` field containing the model identifier string.

## Usage

```python
from any_llm import list_models

models = list_models("openai")
for model in models:
    print(model.id)
```

### Async

```python
import asyncio
from any_llm import alist_models

async def main():
    models = await alist_models("mistral")
    for model in models:
        print(model.id)

asyncio.run(main())
```

### Using the AnyLLM class

```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai")
models = llm.list_models()
print(f"Available models: {len(models)}")
```

{% hint style="info" %}
Not all providers support listing models. Check the [providers page](../providers.md) for support details, or query `ProviderMetadata.list_models` programmatically.
{% endhint %}
