---
title: Model Types
description: Data models for model operations
---

The `Model` type represents a single model returned by `any_llm.list_models()` and `any_llm.alist_models()`.

## `Model`

Re-exported from `openai.types.model.Model`.

**Import:** `from any_llm.types.model import Model`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | The model identifier (e.g., `"gpt-4.1-mini"`, `"mistral-small-latest"`). |
| `created` | `int` | Unix timestamp (seconds) of when the model was created. |
| `object` | `str` | Always `"model"`. |
| `owned_by` | `str` | The organization that owns the model. |

## Usage

```python
from any_llm import list_models

models = list_models("openai")
for model in models:
    print(f"{model.id} (owned by {model.owned_by})")
```

{% hint style="info" %}
The `Model` type is a direct re-export from the OpenAI SDK. any-llm normalizes all provider responses into this format so you get a consistent interface regardless of which provider you query.
{% endhint %}
