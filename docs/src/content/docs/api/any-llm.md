---
title: AnyLLM
description: The AnyLLM class - provider interface with metadata access and reusability
---

The `AnyLLM` class is the provider interface at the core of any-llm. Use it when you need to make multiple requests against the same provider without re-instantiating on every call.

## Creating an Instance

### `AnyLLM.create()`

Factory method that returns a configured `AnyLLM` instance for the given provider.

```python
@classmethod
def create(
    cls,
    provider: str | LLMProvider,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> AnyLLM
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `provider` | `str \| LLMProvider` | Provider name (e.g., `"openai"`, `"anthropic"`, `"mistral"`) |
| `api_key` | `str \| None` | API key. Falls back to the provider's environment variable if omitted. |
| `api_base` | `str \| None` | Override the provider's default base URL. |
| `**kwargs` | `Any` | Additional provider-specific arguments passed to client initialization. |

**Returns:** An `AnyLLM` instance bound to the specified provider.

```python
from any_llm import AnyLLM

llm = AnyLLM.create("openai", api_key="sk-...")

response = llm.completion(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Static Methods

### `AnyLLM.split_model_provider()`

Parses a combined `"provider:model"` string into its components.

```python
@classmethod
def split_model_provider(cls, model: str) -> tuple[LLMProvider, str]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | Combined identifier in `"provider:model"` format (e.g., `"openai:gpt-4.1-mini"`). The legacy `"provider/model"` format is also accepted but deprecated. |

**Returns:** A `(LLMProvider, model_name)` tuple.

**Raises:** `ValueError` if the string does not contain a `:` or `/` delimiter.

```python
provider, model_name = AnyLLM.split_model_provider("anthropic:claude-sonnet-4-20250514")
# provider = LLMProvider.ANTHROPIC
# model_name = "claude-sonnet-4-20250514"
```

### `AnyLLM.get_all_provider_metadata()`

Returns metadata for every supported provider, sorted alphabetically by name.

```python
@classmethod
def get_all_provider_metadata(cls) -> list[ProviderMetadata]
```

**Returns:** A list of [`ProviderMetadata`](/any-llm/api/types/provider/) objects.

```python
for meta in AnyLLM.get_all_provider_metadata():
    print(f"{meta.name}: streaming={meta.streaming}, embedding={meta.embedding}")
```

### `AnyLLM.get_supported_providers()`

Returns a list of all supported provider key strings.

```python
@classmethod
def get_supported_providers(cls) -> list[str]
```

**Returns:** `list[str]` of provider keys (e.g., `["anthropic", "openai", ...]`).

## Instance Methods

All instance methods below are called on an `AnyLLM` object returned by `AnyLLM.create()`.

### `completion()` / `acompletion()`

Create a chat completion. See the [Completion](/any-llm/api/completion/) reference for the full parameter list.

```python
def completion(self, model, messages, *, stream=None, response_format=None, **kwargs)
    -> ChatCompletion | Iterator[ChatCompletionChunk] | ParsedChatCompletion

async def acompletion(self, model, messages, *, stream=None, response_format=None, **kwargs)
    -> ChatCompletion | AsyncIterator[ChatCompletionChunk] | ParsedChatCompletion
```

### `responses()` / `aresponses()`

Create a response using the OpenResponses API. See the [Responses](/any-llm/api/responses/) reference.

```python
def responses(self, **kwargs)
    -> ResponseResource | Response | Iterator[ResponseStreamEvent]

async def aresponses(self, **kwargs)
    -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]
```

### `messages()` / `amessages()`

Create a message using the Anthropic Messages API format. All providers support this through automatic conversion.

```python
def messages(self, **kwargs)
    -> MessageResponse | Iterator[MessageStreamEvent]

async def amessages(self, model, messages, max_tokens, **kwargs)
    -> MessageResponse | AsyncIterator[MessageStreamEvent]
```

### `list_models()` / `alist_models()`

List available models for this provider. See the [List Models](/any-llm/api/list-models/) reference.

```python
def list_models(self, **kwargs) -> Sequence[Model]
async def alist_models(self, **kwargs) -> Sequence[Model]
```

### `create_batch()` / `acreate_batch()`

Create a batch job. See the [Batch](/any-llm/api/batch/) reference.

```python
def create_batch(self, **kwargs) -> Batch
async def acreate_batch(self, input_file_path, endpoint, completion_window="24h", metadata=None, **kwargs) -> Batch
```

### `get_provider_metadata()`

Returns metadata for this provider instance's class.

```python
@classmethod
def get_provider_metadata(cls) -> ProviderMetadata
```

**Returns:** A [`ProviderMetadata`](/any-llm/api/types/provider/) object describing the provider's capabilities.

```python
llm = AnyLLM.create("mistral")
meta = llm.get_provider_metadata()
print(f"Supports streaming: {meta.streaming}")
print(f"Supports embedding: {meta.embedding}")
print(f"Supports responses: {meta.responses}")
```
