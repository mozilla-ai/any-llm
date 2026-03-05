---
title: Completion Types
description: Data models and types for completion operations
---

The completion types used by `any_llm.completion()` and `any_llm.acompletion()` are re-exports from the [OpenAI Python SDK](https://github.com/openai/openai-python), extended where needed to support additional fields like reasoning content.

## Primary Types

### `ChatCompletion`

The response object for a non-streaming completion request. Extends `openai.types.chat.ChatCompletion` with support for reasoning content in the message choices.

**Import:** `from any_llm.types.completion import ChatCompletion`

Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique completion identifier. |
| `choices` | `list[Choice]` | List of completion choices. Each choice has a `message` with `content`, `role`, and optionally `reasoning` and `tool_calls`. |
| `model` | `str` | The model used. |
| `usage` | `CompletionUsage \| None` | Token usage (prompt, completion, total). |

### `ChatCompletionChunk`

A single chunk in a streaming completion response. Extends `openai.types.chat.ChatCompletionChunk`.

**Import:** `from any_llm.types.completion import ChatCompletionChunk`

Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Completion identifier (same across all chunks). |
| `choices` | `list[ChunkChoice]` | Each chunk choice has a `delta` with incremental `content`, `role`, and optionally `reasoning`. |
| `model` | `str` | The model used. |

### `ChatCompletionMessage`

A message within a completion response. Extends `openai.types.chat.ChatCompletionMessage` with a `reasoning` field.

**Import:** `from any_llm.types.completion import ChatCompletionMessage`

| Field | Type | Description |
|-------|------|-------------|
| `role` | `str` | Message role (e.g., `"assistant"`). |
| `content` | `str \| None` | Text content of the message. |
| `reasoning` | `Reasoning \| None` | Reasoning/thinking content (when the model supports it). |
| `tool_calls` | `list[ChatCompletionMessageToolCall] \| None` | Tool calls requested by the model. |
| `annotations` | `list[dict] \| None` | Annotations attached to the message. |

### `ParsedChatCompletion`

Returned when `response_format` is a Pydantic `BaseModel` subclass or a dataclass type. Extends `ChatCompletion` with a generic type parameter.

**Import:** `from any_llm import ParsedChatCompletion`

Access the parsed object via `response.choices[0].message.parsed`, which will be an instance of the type passed as `response_format`.

### `CreateEmbeddingResponse`

Response object for embedding requests. Re-exported directly from `openai.types.CreateEmbeddingResponse`.

**Import:** `from any_llm.types.completion import CreateEmbeddingResponse`

| Field | Type | Description |
|-------|------|-------------|
| `data` | `list[Embedding]` | List of embedding objects, each with an `embedding` vector and `index`. |
| `model` | `str` | The model used. |
| `usage` | `Usage` | Token usage with `prompt_tokens` and `total_tokens`. |

### `ReasoningEffort`

A literal type controlling reasoning depth for models that support it.

**Import:** `from any_llm.types.completion import ReasoningEffort`

```python
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "auto"]
```

The value `"auto"` (the default) maps to each provider's own default reasoning level.

## Additional Re-exports

The following types are also available from `any_llm.types.completion`:

| Type | Origin | Description |
|------|--------|-------------|
| `CompletionUsage` | `openai.types.CompletionUsage` | Token usage counts. |
| `Function` | `openai.types.chat` | Function definition within a tool call. |
| `Embedding` | `openai.types.Embedding` | Single embedding vector with index. |
| `ChoiceDeltaToolCall` | `openai.types.chat` | Tool call delta in streaming chunks. |

For full field-level documentation of the base OpenAI types, see the [OpenAI Python SDK reference](https://github.com/openai/openai-python).
