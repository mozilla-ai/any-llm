---
title: Completion
description: Create chat completions with any provider
---

The `completion` and `acompletion` functions are the primary way to generate chat completions across all supported providers. They accept an OpenAI-compatible parameter set and return OpenAI-compatible response types.

## `any_llm.completion()`

```python
def completion(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | type | None = None,
    stream: bool | None = None,
    n: int | None = None,
    stop: str | list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    user: str | None = None,
    parallel_tool_calls: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: dict[str, float] | None = None,
    stream_options: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: ReasoningEffort | None = "auto",
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion | Iterator[ChatCompletionChunk]
```

## `any_llm.acompletion()`

Async variant with the same parameters. Returns `ChatCompletion | AsyncIterator[ChatCompletionChunk]`.

```python
async def acompletion(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    *,
    # ... same keyword arguments as completion() ...
    **kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. **Recommended**: use with a separate `provider` parameter (e.g., `model="gpt-4.1-mini"`). **Alternative**: combined `"provider:model"` format (e.g., `"openai:gpt-4.1-mini"`). |
| `messages` | `list[dict \| ChatCompletionMessage]` | *required* | Conversation messages. Each item is a dict with `role` and `content` keys, or a `ChatCompletionMessage` object. |
| `provider` | `str \| LLMProvider \| None` | `None` | Provider name (e.g., `"openai"`, `"mistral"`). When provided, `model` should contain only the model name. |
| `tools` | `list[dict \| Callable] \| None` | `None` | Tools for tool calling. Accepts Python callables (auto-converted) or OpenAI tool-format dicts. |
| `tool_choice` | `str \| dict \| None` | `None` | Controls which tools the model can call. |
| `temperature` | `float \| None` | `None` | Controls randomness (0.0 to 2.0). |
| `top_p` | `float \| None` | `None` | Nucleus sampling parameter (0.0 to 1.0). |
| `max_tokens` | `int \| None` | `None` | Maximum number of tokens to generate. |
| `response_format` | `dict \| type \| None` | `None` | Response format specification. Pass a Pydantic `BaseModel` subclass or a dataclass type for structured output. |
| `stream` | `bool \| None` | `None` | Set to `True` to receive a streaming iterator of chunks. |
| `n` | `int \| None` | `None` | Number of completions to generate. |
| `stop` | `str \| list[str] \| None` | `None` | Stop sequences that halt generation. |
| `presence_penalty` | `float \| None` | `None` | Penalize tokens based on prior presence in the text. |
| `frequency_penalty` | `float \| None` | `None` | Penalize tokens based on their frequency in the text. |
| `seed` | `int \| None` | `None` | Random seed for reproducible results. |
| `api_key` | `str \| None` | `None` | API key. Falls back to the provider's environment variable. |
| `api_base` | `str \| None` | `None` | Override the provider's base URL. |
| `user` | `str \| None` | `None` | Unique identifier for the end user. |
| `parallel_tool_calls` | `bool \| None` | `None` | Allow parallel tool calls. |
| `logprobs` | `bool \| None` | `None` | Include token-level log probabilities. |
| `top_logprobs` | `int \| None` | `None` | Number of top alternatives to return when `logprobs` is enabled. |
| `logit_bias` | `dict[str, float] \| None` | `None` | Bias the likelihood of specified tokens. |
| `stream_options` | `dict[str, Any] \| None` | `None` | Additional streaming behavior options. |
| `max_completion_tokens` | `int \| None` | `None` | Maximum tokens for the completion (provider-dependent). |
| `reasoning_effort` | `ReasoningEffort \| None` | `"auto"` | Reasoning effort level for models that support it. `"auto"` maps to each provider's default. |
| `client_args` | `dict[str, Any] \| None` | `None` | Provider-specific arguments passed to client initialization. |
| `**kwargs` | `Any` | | Additional provider-specific arguments passed to the API call. |

## Return Value

- **Non-streaming** (`stream=None` or `stream=False`): Returns a [`ChatCompletion`](/any-llm/api/types/completion/) object.
- **Streaming** (`stream=True`): Returns an `Iterator[ChatCompletionChunk]` (sync) or `AsyncIterator[ChatCompletionChunk]` (async).
- **Structured output** (when `response_format` is a Pydantic model or dataclass): Returns a `ParsedChatCompletion[T]` with a `.choices[0].message.parsed` field containing the deserialized object.

## Usage

### Basic completion

```python
from any_llm import completion

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)
print(response.choices[0].message.content)
```

### Streaming

```python
for chunk in completion(
    model="gpt-4.1-mini",
    provider="openai",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

### Async

```python
import asyncio
from any_llm import acompletion

async def main():
    response = await acompletion(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Structured output

```python
from pydantic import BaseModel
from any_llm import completion

class CityInfo(BaseModel):
    name: str
    country: str
    population: int

response = completion(
    model="gpt-4.1-mini",
    provider="openai",
    messages=[{"role": "user", "content": "Tell me about Paris."}],
    response_format=CityInfo,
)
city = response.choices[0].message.parsed
print(f"{city.name}, {city.country} - pop. {city.population}")
```

### Tool calling

```python
from any_llm import completion

def get_weather(location: str, unit: str = "F") -> str:
    """Get weather information for a location.

    Args:
        location: The city or location to get weather for
        unit: Temperature unit, either 'C' or 'F'

    Returns:
        Current weather description
    """
    return f"Weather in {location} is sunny and 75{unit}!"

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[get_weather],
)
```
