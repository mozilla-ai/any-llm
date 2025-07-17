from typing import Any

from pydantic import BaseModel

from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.types import ChatCompletion
from any_llm.providers.deepseek.utils import _convert_pydantic_to_deepseek_json


class DeepseekProvider(BaseOpenAIProvider):
    """
    DeepSeek Provider implementation.

    This provider connects to DeepSeek's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use DeepSeek's configuration.

    Configuration:
    - api_key: DeepSeek API key (can be set via DEEPSEEK_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to DeepSeek's API)

    Example usage:
        config = ApiConfig(api_key="your-deepseek-api-key")
        provider = DeepseekProvider(config)
        response = provider.completion("deepseek-chat", messages=[...])
    """

    # DeepSeek-specific configuration
    DEFAULT_API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    PROVIDER_NAME = "DeepSeek"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize DeepSeek provider with DeepSeek configuration."""
        super().__init__(config)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Create a chat completion using DeepSeek with Pydantic model support.

        DeepSeek doesn't support Pydantic parsing directly, so we convert
        Pydantic models to JSON format instructions.
        """
        # Handle Pydantic model conversion for DeepSeek
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to DeepSeek JSON format
                modified_messages = _convert_pydantic_to_deepseek_json(response_format, messages)
                kwargs["response_format"] = {"type": "json_object"}
                messages = modified_messages

        # Call the parent completion method
        return super().completion(model, messages, **kwargs)
