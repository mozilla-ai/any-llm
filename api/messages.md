---
title: Messages
description: Anthropic Messages API for all providers
---

The `messages` and `amessages` functions use the Anthropic Messages API format. All providers support this through automatic conversion, so you can use the same Anthropic-style message format regardless of backend.

## `any_llm.messages()`

```
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
    prompt_cache_key: str | None = None,
    context_management: dict[str, Any] | None = None,
    betas: list[str] | None = None,
    output_format: type | dict[str, Any] | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MessageResponse | ParsedMessage[Any] | ParsedBetaMessage[Any] | Iterator[MessageStartEvent | MessageDeltaEvent | MessageStopEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent]
```

## `any_llm.amessages()`

Async variant with the same parameters. Returns `MessageResponse | AsyncIterator[MessageStreamEvent]`.

```
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
    prompt_cache_key: str | None = None,
    context_management: dict[str, Any] | None = None,
    betas: list[str] | None = None,
    output_format: type | dict[str, Any] | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MessageResponse | ParsedMessage[Any] | ParsedBetaMessage[Any] | AsyncIterator[MessageStartEvent | MessageDeltaEvent | MessageStopEvent | ContentBlockStartEvent | ContentBlockDeltaEvent | ContentBlockStopEvent]
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
| `prompt_cache_key` | `str \| None` | None | A key to use when reading from or writing to a provider's prompt cache. |
| `context_management` | `dict[str, Any] \| None` | None | Anthropic context management configuration. The `compact_20260112` strategy requires a supported model. Its `input_tokens` trigger value must be at least 50,000 when provided; see [Anthropic's compaction documentation](https://platform.claude.com/docs/en/build-with-claude/compaction). |
| `betas` | `list[str] \| None` | None | Anthropic beta identifiers. |
| `output_format` | `type \| dict[str, Any] \| None` | None | Structured output, mirroring Anthropic's ``messages.parse``/``output_config``. Either a Pydantic ``BaseModel``/dataclass **type** (typed ``parsed_output``) or a raw Anthropic ``output_config`` **dict** for non-Pydantic JSON schemas (``parsed_output`` holds the parsed JSON). The call returns Anthropic's ``ParsedMessage``. Not supported with streaming. |
| `timeout` | `float \| None` | None | Per-request timeout in seconds, passed through to the provider's client/SDK. An explicit ``None`` is treated the same as omitting it (the provider's default applies), so it cannot request an unbounded timeout. Providers that have no per-request timeout raise `UnsupportedParameterError`; set a timeout on their client via `client_args` instead. |
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
