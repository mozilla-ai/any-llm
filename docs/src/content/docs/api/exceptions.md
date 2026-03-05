---
title: Exceptions
description: Unified exception hierarchy for all providers
---

any-llm provides a unified exception hierarchy so you can handle errors consistently regardless of which provider is being used. When unified exceptions are enabled, provider-specific SDK errors are automatically mapped to the appropriate any-llm exception type.

:::note[Opt-in Feature]
Unified exception handling is opt-in. Set the `ANY_LLM_UNIFIED_EXCEPTIONS=1` environment variable to enable automatic conversion from provider-specific exceptions.
:::

## Exception Hierarchy

All exceptions inherit from `AnyLLMError`:

```
AnyLLMError
├── RateLimitError
├── AuthenticationError
├── InvalidRequestError
├── ProviderError
├── ContentFilterError
├── ModelNotFoundError
├── ContextLengthExceededError
├── MissingApiKeyError
├── UnsupportedProviderError
└── UnsupportedParameterError
```

## `AnyLLMError`

Base exception for all any-llm errors. Preserves the original provider exception for debugging.

```python
class AnyLLMError(Exception):
    def __init__(
        self,
        message: str | None = None,
        original_exception: Exception | None = None,
        provider_name: str | None = None,
    ) -> None: ...
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error message. |
| `original_exception` | `Exception \| None` | The original SDK exception that triggered this error. |
| `provider_name` | `str \| None` | Name of the provider that raised the error (if available). |

The string representation includes the provider name when available: `"[openai] Rate limit exceeded"`.

## Provider Errors

### `RateLimitError`

Raised when the API rate limit is exceeded.

```python
class RateLimitError(AnyLLMError): ...
```

Default message: `"Rate limit exceeded"`

### `AuthenticationError`

Raised when authentication with the provider fails (invalid or missing API key).

```python
class AuthenticationError(AnyLLMError): ...
```

Default message: `"Authentication failed"`

### `InvalidRequestError`

Raised when the request to the provider is malformed or contains invalid parameters.

```python
class InvalidRequestError(AnyLLMError): ...
```

Default message: `"Invalid request"`

### `ProviderError`

Raised when the provider encounters an internal error (5xx-class errors).

```python
class ProviderError(AnyLLMError): ...
```

Default message: `"Provider error"`

### `ContentFilterError`

Raised when content is blocked by the provider's safety filter.

```python
class ContentFilterError(AnyLLMError): ...
```

Default message: `"Content blocked by safety filter"`

### `ModelNotFoundError`

Raised when the requested model is not found or not available.

```python
class ModelNotFoundError(AnyLLMError): ...
```

Default message: `"Model not found"`

### `ContextLengthExceededError`

Raised when the input exceeds the model's maximum context length.

```python
class ContextLengthExceededError(AnyLLMError): ...
```

Default message: `"Context length exceeded"`

## Configuration Errors

### `MissingApiKeyError`

Raised when a required API key is not provided via the parameter or environment variable.

```python
class MissingApiKeyError(AnyLLMError):
    def __init__(self, provider_name: str, env_var_name: str) -> None: ...
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `provider_name` | `str` | Name of the provider requiring the key. |
| `env_var_name` | `str` | Environment variable name that was checked. |

Example message: `"No openai API key provided. Please provide it in the config or set the OPENAI_API_KEY environment variable."`

### `UnsupportedProviderError`

Raised when an unsupported provider is specified.

```python
class UnsupportedProviderError(AnyLLMError):
    def __init__(self, provider_key: str, supported_providers: list[str]) -> None: ...
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `provider_key` | `str` | The unsupported provider key that was specified. |
| `supported_providers` | `list[str]` | List of valid provider keys. |

### `UnsupportedParameterError`

Raised when a parameter is not supported by the provider.

```python
class UnsupportedParameterError(AnyLLMError):
    def __init__(self, parameter_name: str, provider_name: str, additional_message: str | None = None) -> None: ...
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `parameter_name` | `str` | The unsupported parameter name. |
| `provider_name` | `str` | Name of the provider (also accessible via the inherited `provider_name` attribute). |

## Usage

```python
from any_llm import completion
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    RateLimitError,
    ContextLengthExceededError,
)

try:
    response = completion(
        model="gpt-4.1-mini",
        provider="openai",
        messages=[{"role": "user", "content": "Hello!"}],
    )
except RateLimitError as e:
    print(f"Rate limited by {e.provider_name}: {e.message}")
    # Access the original provider exception for details
    print(f"Original: {e.original_exception}")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except ContextLengthExceededError as e:
    print(f"Input too long: {e.message}")
except AnyLLMError as e:
    # Catch-all for any other any-llm error
    print(f"Error: {e}")
```
