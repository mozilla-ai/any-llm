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
| `choices` | `list[Choice]` |  |

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

## Internal Types

### `CompletionParams`

Normalized parameters for chat completions, used internally to pass structured parameters from the public API to provider implementations.

**Import:** `from any_llm.types.completion import CompletionParams`

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Model identifier (e.g., 'mistral-small-latest') |
| `messages` | `list[dict[str, Any]]` | List of messages for the conversation |
| `tools` | `list[dict[str, Any] \| Any] \| None` | List of tools for tool calling. Should be converted to OpenAI tool format dicts |
| `tool_choice` | `str \| dict[str, Any] \| None` | Controls which tools the model can call |
| `temperature` | `float \| None` | Controls randomness in the response (0.0 to 2.0) |
| `top_p` | `float \| None` | Controls diversity via nucleus sampling (0.0 to 1.0) |
| `max_tokens` | `int \| None` | Maximum number of tokens to generate |
| `response_format` | `dict[str, Any] \| type \| None` | Format specification for the response. Accepts Pydantic BaseModel subclasses, dataclass types, or dicts. |
| `stream` | `bool \| None` | Whether to stream the response |
| `n` | `int \| None` | Number of completions to generate |
| `stop` | `str \| list[str] \| None` | Stop sequences for generation |
| `presence_penalty` | `float \| None` | Penalize new tokens based on presence in text |
| `frequency_penalty` | `float \| None` | Penalize new tokens based on frequency in text |
| `seed` | `int \| None` | Random seed for reproducible results |
| `user` | `str \| None` | Unique identifier for the end user |
| `parallel_tool_calls` | `bool \| None` | Whether to allow parallel tool calls |
| `logprobs` | `bool \| None` | Include token-level log probabilities in the response |
| `top_logprobs` | `int \| None` | Number of top alternatives to return when logprobs are requested |
| `logit_bias` | `dict[str, float] \| None` | Bias the likelihood of specified tokens during generation |
| `stream_options` | `dict[str, Any] \| None` | Additional options controlling streaming behavior |
| `max_completion_tokens` | `int \| None` | Maximum number of tokens for the completion (provider-dependent) |
| `reasoning_effort` | `Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'auto'] \| None` |  |

## Additional Re-exports

The following types are also available from `any_llm.types.completion`:

| Type | Origin | Description |
|------|--------|-------------|
| `CompletionUsage` | `openai.types.CompletionUsage` | Token usage counts. |
| `Function` | `openai.types.chat` | Function definition within a tool call. |
| `Embedding` | `openai.types.Embedding` | Single embedding vector with index. |
| `ChoiceDeltaToolCall` | `openai.types.chat` | Tool call delta in streaming chunks. |

For full field-level documentation of the base OpenAI types, see the [OpenAI Python SDK reference](https://github.com/openai/openai-python).
