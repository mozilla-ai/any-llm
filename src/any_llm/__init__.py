from importlib.metadata import PackageNotFoundError, version

from any_llm.any_llm import AnyLLM
from any_llm.api import acompletion, aembedding, alist_models, aresponses, completion, embedding, list_models, responses
from any_llm.constants import LLMProvider
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
    UnsupportedProviderError,
)

try:
    __version__ = version("any-llm-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


__all__ = [
    "AnyLLM",
    "AnyLLMError",
    "AuthenticationError",
    "ContentFilterError",
    "ContextLengthExceededError",
    "InvalidRequestError",
    "LLMProvider",
    "MissingApiKeyError",
    "ModelNotFoundError",
    "ProviderError",
    "RateLimitError",
    "UnsupportedParameterError",
    "UnsupportedProviderError",
    "acompletion",
    "aembedding",
    "alist_models",
    "aresponses",
    "completion",
    "embedding",
    "list_models",
    "responses",
]
