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
| `name` | `str` | Provider identifier (e.g., `"openai"`, `"anthropic"`). Matches the provider directory name. |
| `env_key` | `str` | Environment variable name for the API key (e.g., `"OPENAI_API_KEY"`). |
| `env_api_base` | `str \| None` | Environment variable for overriding the API base URL, if supported. |
| `doc_url` | `str` | Link to the provider's documentation. |
| `class_name` | `str` | Internal Python class name (e.g., `"OpenaiProvider"`). |
| `streaming` | `bool` | Whether the provider supports streaming completions. |
| `image` | `bool` | Whether the provider supports image inputs in completions. |
| `pdf` | `bool` | Whether the provider supports PDF inputs in completions. |
| `embedding` | `bool` | Whether the provider supports the Embedding API. |
| `reasoning` | `bool` | Whether the provider supports reasoning/thinking traces. |
| `responses` | `bool` | Whether the provider supports the Responses API. |
| `completion` | `bool` | Whether the provider supports the Completion API. |
| `messages` | `bool` | Whether the provider supports the Messages API. |
| `list_models` | `bool` | Whether the provider supports listing available models. |
| `batch_completion` | `bool` | Whether the provider supports the Batch API. |

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
