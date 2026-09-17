from typing import ClassVar

from any_llm.types.files import FileOperation

from .base import BaseOpenAIProvider
from .files import OpenAIFileMethods


class OpenaiProvider(OpenAIFileMethods, BaseOpenAIProvider):
    API_BASE = "https://api.openai.com/v1"

    ENV_API_KEY_NAME = "OPENAI_API_KEY"
    ENV_API_BASE_NAME = "OPENAI_BASE_URL"
    PROVIDER_NAME = "openai"
    PROVIDER_DOCUMENTATION_URL = "https://platform.openai.com/docs/api-reference"
    PROMPT_CACHE_KEY_SUPPORT = "supported"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True
    SUPPORTS_IMAGE_GENERATION = True
    SUPPORTS_AUDIO_TRANSCRIPTION = True
    SUPPORTS_AUDIO_SPEECH = True
    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]] = frozenset(
        {"upload", "list", "retrieve", "download", "delete"}
    )
