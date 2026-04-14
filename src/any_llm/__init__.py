from importlib.metadata import PackageNotFoundError, version

from any_llm.any_llm import AnyLLM
from any_llm.api import (
    acompletion,
    aembedding,
    alist_models,
    amessages,
    aresponses,
    aspeech,
    atranscription,
    completion,
    embedding,
    list_models,
    messages,
    responses,
    speech,
    transcription,
)
from any_llm.constants import LLMProvider
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContentFilterFinishReasonError,
    ContextLengthExceededError,
    InvalidRequestError,
    LengthFinishReasonError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
    UnsupportedProviderError,
)
from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams, Transcription, TranscriptionVerbose
from any_llm.types.completion import ParsedChatCompletion, ParsedChatCompletionMessage, ParsedChoice

try:
    __version__ = version("any-llm-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


__all__ = [
    "AnyLLM",
    "AnyLLMError",
    "AudioSpeechParams",
    "AudioTranscriptionParams",
    "AuthenticationError",
    "ContentFilterError",
    "ContentFilterFinishReasonError",
    "ContextLengthExceededError",
    "InvalidRequestError",
    "LLMProvider",
    "LengthFinishReasonError",
    "MissingApiKeyError",
    "ModelNotFoundError",
    "ParsedChatCompletion",
    "ParsedChatCompletionMessage",
    "ParsedChoice",
    "ProviderError",
    "RateLimitError",
    "Transcription",
    "TranscriptionVerbose",
    "UnsupportedParameterError",
    "UnsupportedProviderError",
    "acompletion",
    "aembedding",
    "alist_models",
    "amessages",
    "aresponses",
    "aspeech",
    "atranscription",
    "completion",
    "embedding",
    "list_models",
    "messages",
    "responses",
    "speech",
    "transcription",
]
