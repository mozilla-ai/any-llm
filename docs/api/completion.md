---
title: Completion
description: Create chat completions with any provider
---

The `completion` and `acompletion` functions are the primary way to generate chat completions across all supported providers. They accept an OpenAI-compatible parameter set and return OpenAI-compatible response types.

## `any_llm.completion()`

```
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
    session_label: str | None = None,
    parallel_tool_calls: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: dict[str, float] | None = None,
    stream_options: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'auto'] | None = "auto",
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion | Iterator[ChatCompletionChunk]
```

## `any_llm.acompletion()`

Async variant with the same parameters. Returns `ChatCompletion | AsyncIterator[ChatCompletionChunk]`.

```
async def acompletion(
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
    session_label: str | None = None,
    parallel_tool_calls: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: dict[str, float] | None = None,
    stream_options: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'auto'] | None = "auto",
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *required* | Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='gpt-4', provider='openai'). **Alternative**: Combined format 'provider:model' (e.g., 'openai:gpt-4'). Legacy format 'provider/model' is also supported but deprecated. |
| `messages` | `list[dict[str, Any] \| ChatCompletionMessage]` | *required* | List of messages for the conversation |
| `provider` | `str \| LLMProvider \| None` | None | **Recommended**: Provider name to use for the request (e.g., 'openai', 'mistral'). When provided, the model parameter should contain only the model name. |
| `tools` | `list[dict[str, Any] \| Callable[..., Any]] \| None` | None | List of tools for tool calling. Can be Python callables or OpenAI tool format dicts |
| `tool_choice` | `str \| dict[str, Any] \| None` | None | Controls which tools the model can call |
| `temperature` | `float \| None` | None | Controls randomness in the response (0.0 to 2.0) |
| `top_p` | `float \| None` | None | Controls diversity via nucleus sampling (0.0 to 1.0) |
| `max_tokens` | `int \| None` | None | Maximum number of tokens to generate |
| `response_format` | `dict[str, Any] \| type \| None` | None | Format specification for the response |
| `stream` | `bool \| None` | None | Whether to stream the response |
| `n` | `int \| None` | None | Number of completions to generate |
| `stop` | `str \| list[str] \| None` | None | Stop sequences for generation |
| `presence_penalty` | `float \| None` | None | Penalize new tokens based on presence in text |
| `frequency_penalty` | `float \| None` | None | Penalize new tokens based on frequency in text |
| `seed` | `int \| None` | None | Random seed for reproducible results |
| `api_key` | `str \| None` | None | API key for the provider |
| `api_base` | `str \| None` | None | Base URL for the provider API |
| `user` | `str \| None` | None | Unique identifier for the end user |
| `session_label` | `str \| None` | None | Optional user session label metadata for platform traces; exported as anyllm.user_session_label |
| `parallel_tool_calls` | `bool \| None` | None | Whether to allow parallel tool calls |
| `logprobs` | `bool \| None` | None | Include token-level log probabilities in the response |
| `top_logprobs` | `int \| None` | None | Number of alternatives to return when logprobs are requested |
| `logit_bias` | `dict[str, float] \| None` | None | Bias the likelihood of specified tokens during generation |
| `stream_options` | `dict[str, Any] \| None` | None | Additional options controlling streaming behavior |
| `max_completion_tokens` | `int \| None` | None | Maximum number of tokens for the completion |
| `reasoning_effort` | `Literal['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'auto'] \| None` | "auto" | Reasoning effort level for models that support it. "auto" will map to each provider's default. |
| `client_args` | `dict[str, Any] \| None` | None | Additional provider-specific arguments that will be passed to the provider's client instantiation. |
| `**kwargs` | `Any` | *required* | Additional provider-specific arguments that will be passed to the provider's API call. |

## Return Value

- **Non-streaming** (`stream=None` or `stream=False`): Returns a [`ChatCompletion`](types/completion.md) object.
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
