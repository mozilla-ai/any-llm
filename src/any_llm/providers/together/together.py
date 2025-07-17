import os
from typing import Any

try:
    import together
    from together.types import (
        ChatCompletionResponse,
    )
    import instructor
except ImportError:
    msg = "together or instructor is not installed. Please install it with `pip install any-llm-sdk[together]`"
    raise ImportError(msg)


from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.helpers import create_completion_from_response


class TogetherProvider(Provider):
    """
    Together AI Provider implementation using the official Together AI SDK with instructor support.

    This provider connects to Together AI's API using the Together SDK with instructor
    handling structured outputs for Pydantic models.

    Configuration:
    - api_key: Together AI API key (can be set via TOGETHER_API_KEY environment variable)
    - api_base: Custom base URL (optional, defaults to Together AI's API)

    Example usage:
        config = ApiConfig(api_key="your-together-api-key")
        provider = TogetherProvider(config)
        response = provider.completion("meta-llama/Llama-3.1-8B-Instruct-Turbo", messages=[...])
    """

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Together AI provider."""
        if not config.api_key:
            config.api_key = os.getenv("TOGETHER_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Together AI", "TOGETHER_API_KEY")

        # Initialize Together client
        if config.api_base:
            self.client = together.Together(api_key=config.api_key, base_url=config.api_base)
        else:
            self.client = together.Together(api_key=config.api_key)

        # Create instructor client for structured output support
        # Together is OpenAI-compatible, so we can use instructor.from_openai
        self.instructor_client = instructor.patch(self.client, mode=instructor.Mode.JSON)  # type: ignore [call-overload]

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Together AI with instructor support for structured outputs."""

        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", "Together")

        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")

            # Use instructor for structured output
            instructor_response = self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_format,
                **kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "together")

        # Make the API call. Since streaming is not supported, this won't be an iter
        response: ChatCompletionResponse = self.client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,
            **kwargs,
        )

        # Convert to OpenAI format using the utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            model=model,
            provider_name="together",
            finish_reason_mapping={
                "stop": "stop",
                "length": "length",
                "tool_calls": "tool_calls",
                "content_filter": "content_filter",
            },
            token_field_mapping={
                "prompt_tokens": "prompt_tokens",
                "completion_tokens": "completion_tokens",
                "total_tokens": "total_tokens",
            },
        )
