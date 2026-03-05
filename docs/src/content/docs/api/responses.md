---
title: Responses
description: OpenResponses API for agentic AI systems
---

The `responses` and `aresponses` functions implement the [OpenResponses specification](https://github.com/openresponsesspec/openresponses), a vendor-neutral API for agentic AI systems. This API supports multi-turn conversations, tool use, and streaming events.

## Return Types

The return type depends on the provider and whether streaming is enabled:

| Condition | Return Type |
|-----------|-------------|
| OpenResponses-compliant provider (non-streaming) | `openresponses_types.ResponseResource` |
| OpenAI-native provider (non-streaming) | `openai.types.responses.Response` |
| Streaming (`stream=True`) | `Iterator[ResponseStreamEvent]` (sync) or `AsyncIterator[ResponseStreamEvent]` (async) |

## `any_llm.responses()`

```python
def responses(
    model: str,
    input_data: str | ResponseInputParam,
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    max_tool_calls: int | None = None,
    parallel_tool_calls: int | None = None,
    reasoning: Any | None = None,
    text: Any | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    truncation: str | None = None,
    store: bool | None = None,
    service_tier: str | None = None,
    user: str | None = None,
    metadata: dict[str, str] | None = None,
    previous_response_id: str | None = None,
    include: list[str] | None = None,
    background: bool | None = None,
    safety_identifier: str | None = None,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    conversation: str | dict[str, Any] | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ResponseResource | Response | Iterator[ResponseStreamEvent]
```

## `any_llm.aresponses()`

Async variant with the same parameters. Returns `ResponseResource | Response | AsyncIterator[ResponseStreamEvent]`.

```python
async def aresponses(
    model: str,
    input_data: str | ResponseInputParam,
    *,
    # ... same keyword arguments as responses() ...
    **kwargs: Any,
) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. Use with a separate `provider` parameter, or combined `"provider:model"` format. |
| `input_data` | `str \| ResponseInputParam` | *required* | Input payload. Can be a plain string or a structured `ResponseInputParam` (list of message items). |
| `provider` | `str \| LLMProvider \| None` | `None` | Provider name (e.g., `"openai"`, `"mistral"`). |
| `tools` | `list[dict \| Callable] \| None` | `None` | Tools for tool calling. Accepts Python callables or OpenAI tool-format dicts. |
| `tool_choice` | `str \| dict \| None` | `None` | Controls which tools the model can call. |
| `max_output_tokens` | `int \| None` | `None` | Maximum number of output tokens to generate. |
| `temperature` | `float \| None` | `None` | Controls randomness (0.0 to 2.0). |
| `top_p` | `float \| None` | `None` | Nucleus sampling parameter (0.0 to 1.0). |
| `stream` | `bool \| None` | `None` | Set to `True` to receive streaming events. |
| `api_key` | `str \| None` | `None` | API key. Falls back to the provider's environment variable. |
| `api_base` | `str \| None` | `None` | Override the provider's base URL. |
| `instructions` | `str \| None` | `None` | System (or developer) message inserted into the model's context. |
| `max_tool_calls` | `int \| None` | `None` | Maximum total built-in tool calls allowed in a response. |
| `parallel_tool_calls` | `int \| None` | `None` | Whether to allow the model to run tool calls in parallel. |
| `reasoning` | `Any \| None` | `None` | Configuration options for reasoning models. |
| `text` | `Any \| None` | `None` | Configuration for text response format (plain text or structured JSON). |
| `presence_penalty` | `float \| None` | `None` | Penalize tokens based on prior presence. |
| `frequency_penalty` | `float \| None` | `None` | Penalize tokens based on frequency. |
| `truncation` | `str \| None` | `None` | Controls input truncation when exceeding the context window. |
| `store` | `bool \| None` | `None` | Whether to store the response for later retrieval. |
| `service_tier` | `str \| None` | `None` | Service tier for the request. |
| `user` | `str \| None` | `None` | Unique end-user identifier. |
| `metadata` | `dict[str, str] \| None` | `None` | Custom metadata key-value pairs (up to 16). |
| `previous_response_id` | `str \| None` | `None` | ID of a prior response to use as the previous turn. |
| `include` | `list[str] \| None` | `None` | Items to include in the response (e.g., `"reasoning.encrypted_content"`). |
| `background` | `bool \| None` | `None` | Run the request in the background and return immediately. |
| `safety_identifier` | `str \| None` | `None` | Stable identifier for safety monitoring. |
| `prompt_cache_key` | `str \| None` | `None` | Key for reading/writing the prompt cache. |
| `prompt_cache_retention` | `str \| None` | `None` | How long to retain a prompt cache entry. |
| `conversation` | `str \| dict \| None` | `None` | Conversation to associate with this response (ID string or conversation object). |
| `client_args` | `dict[str, Any] \| None` | `None` | Provider-specific arguments for client initialization. |
| `**kwargs` | `Any` | | Additional provider-specific arguments for the API call. |

## Usage

### Basic response

```python
from any_llm import responses

result = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="What is the capital of France?",
)
print(result.output_text)
```

### With instructions

```python
result = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="Translate to French: Hello, how are you?",
    instructions="You are a professional translator. Always respond with only the translation.",
)
```

### Streaming

```python
for event in responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="Tell me a short story.",
    stream=True,
):
    print(event)
```

### Multi-turn with `previous_response_id`

```python
first = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="My name is Alice.",
    store=True,
)

second = responses(
    model="gpt-4.1-mini",
    provider="openai",
    input_data="What is my name?",
    previous_response_id=first.id,
)
```

:::note
Not all providers support the Responses API. Check the [providers page](/any-llm/providers/) for support details, or query `ProviderMetadata.responses` programmatically.
:::
