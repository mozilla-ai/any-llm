---
title: Messages Types
description: Data models for the Anthropic Messages API
---

The Messages API types are Pydantic models used by `any_llm.api.messages()` and `any_llm.api.amessages()`.

## Primary Types

### `MessageResponse`

Full response from the Messages API.

**Import:** `from any_llm.types.messages import MessageResponse`


### `MessageContentBlock`

Content block in a Messages API response.

**Import:** `from any_llm.types.messages import MessageContentBlock`


### `MessageUsage`

Token usage information for Messages API.

**Import:** `from any_llm.types.messages import MessageUsage`

| Field | Type | Description |
|-------|------|-------------|
| `cache_creation` | `CacheCreation \| None` | Breakdown of cached tokens by TTL |
| `cache_creation_input_tokens` | `int \| None` | The number of input tokens used to create the cache entry. |
| `cache_read_input_tokens` | `int \| None` | The number of input tokens read from the cache. |
| `inference_geo` | `str \| None` | The geographic region where inference was performed for this request. |
| `input_tokens` | `int` | The number of input tokens which were used. |
| `output_tokens` | `int` | The number of output tokens which were used. |
| `server_tool_use` | `ServerToolUsage \| None` | The number of server tool requests. |
| `service_tier` | `Literal['standard', 'priority', 'batch'] \| None` | If the request used the priority, standard, or batch tier. |

### `MessageStreamEvent`

Union of Anthropic SDK stream event types, re-exported from the `anthropic` package:

- `MessageStartEvent` — `type: 'message_start'`, `message: Message`
- `MessageDeltaEvent` — `type: 'message_delta'`, `delta: Delta`, `usage: MessageDeltaUsage`
- `MessageStopEvent` — `type: 'message_stop'`
- `ContentBlockStartEvent` — `type: 'content_block_start'`, `index: int`, `content_block: ContentBlock`
- `ContentBlockDeltaEvent` — `type: 'content_block_delta'`, `index: int`, `delta: RawContentBlockDelta`
- `ContentBlockStopEvent` — `type: 'content_block_stop'`, `index: int`

**Import:** `from any_llm.types.messages import MessageStreamEvent`


## Internal Types

### `MessagesParams`

Normalized parameters for the Anthropic Messages API, used internally to pass structured parameters from the public API to provider implementations.

**Import:** `from any_llm.types.messages import MessagesParams`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model identifier |
| `messages` | `list[dict[str, Any]]` | List of messages for the conversation |
| `max_tokens` | `int` | Maximum number of tokens to generate (required by Anthropic API) |
| `system` | `str \| list[dict[str, Any]] \| None` | System prompt (string or list of content blocks with optional cache_control) |
| `temperature` | `float \| None` | Controls randomness in the response (0.0 to 1.0) |
| `top_p` | `float \| None` | Controls diversity via nucleus sampling |
| `top_k` | `int \| None` | Only sample from the top K options for each subsequent token |
| `stream` | `bool \| None` | Whether to stream the response |
| `stop_sequences` | `list[str] \| None` | Custom text sequences that will cause the model to stop generating |
| `tools` | `list[dict[str, Any]] \| None` | List of tools in Anthropic format ({name, description, input_schema}) |
| `tool_choice` | `dict[str, Any] \| None` | Controls which tool the model uses |
| `metadata` | `dict[str, Any] \| None` | Request metadata |
| `thinking` | `dict[str, Any] \| None` | Thinking/reasoning configuration |
| `cache_control` | `dict[str, Any] \| None` | Cache control configuration for prompt caching |
