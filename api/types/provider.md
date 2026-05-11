---
title: Provider Types
description: Data models for provider operations
---

The `ProviderMetadata` type describes a provider's capabilities and configuration. It is returned by `AnyLLM.get_provider_metadata()` and `AnyLLM.get_all_provider_metadata()`.

## `ProviderMetadata`

A Pydantic `BaseModel` containing provider information and feature flags.

**Import:** `from any_llm.types.provider import ProviderMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` |  |
| `env_key` | `str` |  |
| `env_api_base` | `str \| None` |  |
| `doc_url` | `str` |  |
| `streaming` | `bool` |  |
| `reasoning` | `bool` |  |
| `completion` | `bool` |  |
| `embedding` | `bool` |  |
| `responses` | `bool` |  |
| `image` | `bool` |  |
| `pdf` | `bool` |  |
| `class_name` | `str` |  |
| `list_models` | `bool` |  |
| `messages` | `bool` |  |
| `batch_completion` | `bool` |  |
| `image_generation` | `bool` |  |

## Usage

### Single provider

```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai")
meta = llm.get_provider_metadata()

print(f"Provider: {meta.name}")
print(f"API key env var: {meta.env_key}")
print(f"Supports streaming: {meta.streaming}")
print(f"Supports embedding: {meta.embedding}")
print(f"Supports responses: {meta.responses}")
```

### All providers

```python
from any_llm import AnyLLM

for meta in AnyLLM.get_all_provider_metadata():
    features = []
    if meta.streaming:
        features.append("streaming")
    if meta.embedding:
        features.append("embedding")
    if meta.reasoning:
        features.append("reasoning")
    if meta.responses:
        features.append("responses")
    print(f"{meta.name}: {', '.join(features) or 'completion only'}")
```
