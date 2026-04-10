---
title: Messages
description: Anthropic Messages API for all providers
---

The `messages` and `amessages` functions use the Anthropic Messages API format. All providers support this through automatic conversion, so you can use the same Anthropic-style message format regardless of backend.

## `any_llm.messages()`

```python
def messages(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    provider: str | LLMProvider | None = None,
    system: str | list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stream: bool | None = None,
    stop_sequences: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    thinking: dict[str, Any] | None = None,
    cache_control: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MessageResponse | Iterator[RawMessageStartEvent | RawMessageDeltaEvent | RawMessageStopEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawContentBlockStopEvent]
```

## `any_llm.amessages()`

Async variant with the same parameters. Returns `MessageResponse | AsyncIterator[MessageStreamEvent]`.

```python
async def amessages(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    provider: str | LLMProvider | None = None,
    system: str | list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stream: bool | None = None,
    stop_sequences: list[str] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    thinking: dict[str, Any] | None = None,
    cache_control: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MessageResponse | AsyncIterator[RawMessageStartEvent | RawMessageDeltaEvent | RawMessageStopEvent | RawContentBlockStartEvent | RawContentBlockDeltaEvent | RawContentBlockStopEvent]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. **Recommended**: Use with separate `provider` parameter. **Alternative**: Combined format 'provider:model'. |
| `messages` | `list[dict[str, Any]]` | *required* | List of messages in Anthropic format. |
| `max_tokens` | `int` | *required* | Maximum number of tokens to generate. |
| `provider` | `str \| LLMProvider \| None` | None | Provider name to use for the request. |
| `system` | `str \| list[dict[str, Any]] \| None` | None | System prompt (string or list of content blocks with optional cache_control). |
| `temperature` | `float \| None` | None | Controls randomness (0.0 to 1.0). |
| `top_p` | `float \| None` | None | Controls diversity via nucleus sampling. |
| `top_k` | `int \| None` | None | Only sample from the top K options. |
| `stream` | `bool \| None` | None | Whether to stream the response. |
| `stop_sequences` | `list[str] \| None` | None | Custom stop sequences. |
| `tools` | `list[dict[str, Any]] \| None` | None | List of tools in Anthropic format. |
| `tool_choice` | `dict[str, Any] \| None` | None | Controls which tool the model uses. |
| `metadata` | `dict[str, Any] \| None` | None | Request metadata. |
| `thinking` | `dict[str, Any] \| None` | None | Thinking/reasoning configuration. |
| `cache_control` | `dict[str, Any] \| None` | None | Cache control configuration for prompt caching. |
| `api_key` | `str \| None` | None | API key for the provider. |
| `api_base` | `str \| None` | None | Base URL for the provider API. |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments for client instantiation. |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments. |

## Return Value

- **Non-streaming**: Returns a [`MessageResponse`](types/messages.md) object.
- **Streaming** (`stream=True`): Returns an `Iterator[MessageStreamEvent]` (sync) or `AsyncIterator[MessageStreamEvent]` (async).

## Usage

### Basic message

```python
from any_llm.api import messages

response = messages(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1024,
)
print(response.content[0].text)
```

### With system prompt

```python
response = messages(
    model="claude-sonnet-4-20250514",
    provider="anthropic",
    messages=[{"role": "user", "content": "Translate to French: Hello"}],
    max_tokens=1024,
    system="You are a professional translator.",
)
```

### Async

```python
import asyncio
from any_llm.api import amessages

async def main():
    response = await amessages(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=1024,
    )
    print(response.content[0].text)

asyncio.run(main())
```
