---
title: Model Types
description: Data models for model operations
---

The `Model` type represents a single model returned by `any_llm.list_models()` and `any_llm.alist_models()`.

## `Model`

Extends `openai.types.model.Model` with `context_window`. Any other field a provider sends is kept as an extra attribute.

**Import:** `from any_llm.types.model import Model`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | The model identifier (e.g., `"gpt-4.1-mini"`, `"mistral-small-latest"`). |
| `created` | `int` | Unix timestamp (seconds) of when the model was created. |
| `object` | `str` | Always `"model"`. |
| `owned_by` | `str` | The organization that owns the model. |
| `context_window` | `int \| None` | How many tokens the model's context holds as this provider serves it, or `None` when its listing does not say. Read from `max_model_len` (vLLM, SGLang), `context_length` (OpenRouter, Together) or `context_window`. A self-hosted engine reports the length it was started with, which can be below the model's own maximum. |

## Usage

```python
from any_llm import list_models

models = list_models("openai")
for model in models:
    print(f"{model.id} (owned by {model.owned_by}, context {model.context_window})")
```

{% hint style="info" %}
`Model` is a subclass of the OpenAI SDK's `Model`, so it can be used wherever that type is expected. any-llm normalizes all provider responses into this format so you get a consistent interface regardless of which provider you query.
{% endhint %}
