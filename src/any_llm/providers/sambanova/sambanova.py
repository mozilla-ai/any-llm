from typing import Any

try:
    import instructor
except ImportError:
    msg = "instructor is not installed. Please install it with `pip install any-llm-sdk[sambanova]`"
    raise ImportError(msg)

from openai import OpenAI
from any_llm.types import ChatCompletion
from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

from any_llm.provider import ApiConfig, convert_instructor_response
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.exceptions import UnsupportedParameterError


class SambanovaProvider(BaseOpenAIProvider):
    """
    SambaNova Provider implementation.

    This provider connects to SambaNova's API using the OpenAI-compatible client.
    It extends BaseOpenAIProvider to use SambaNova's configuration.

    Configuration:
    - api_key: SambaNova API key (can be set via SAMBANOVA_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to SambaNova's API)

    Example usage:
        config = ApiConfig(api_key="your-sambanova-api-key")
        provider = SambanovaProvider(config)
        response = provider.completion("your-model", messages=[...])
    """

    # SambaNova-specific configuration
    DEFAULT_API_BASE = "https://api.sambanova.ai/v1/"
    ENV_API_KEY_NAME = "SAMBANOVA_API_KEY"
    PROVIDER_NAME = "SambaNova"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize SambaNova provider with SambaNova configuration."""
        super().__init__(config)
        # Initialize instructor client for structured output
        self._initialize_client(config)
        self._initialize_instructor_client(config)

    def _initialize_instructor_client(self, config: ApiConfig) -> None:
        """Initialize instructor client for structured output."""
        # Create OpenAI client with SambaNova configuration
        openai_client = OpenAI(
            base_url=config.api_base or self.DEFAULT_API_BASE,
            api_key=config.api_key,
        )

        # Wrap with instructor
        self.instructor_client = instructor.from_openai(openai_client)

    def completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Make the API call to SambaNova service with instructor for structured output."""

        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", "sambanova")

        if "response_format" in kwargs:
            # Use instructor for structured output
            response_format = kwargs.pop("response_format")
            response = self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )
            # Convert instructor response to ChatCompletion format
            response = convert_instructor_response(response, model, "sambanova")
        else:
            # Use standard OpenAI client for regular completions
            response = self.client.chat.completions.create(  # type: ignore[return-value]
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
        if isinstance(response, OpenAIChatCompletion):
            return ChatCompletion.model_validate(response.model_dump())
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
