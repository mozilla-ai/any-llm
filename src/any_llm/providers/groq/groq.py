import os
from typing import Any

try:
    import groq
    import instructor
except ImportError:
    msg = "groq or instructor is not installed. Please install it with `pip install any-llm-sdk[groq]`"
    raise ImportError(msg)


from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.helpers import create_completion_from_response


class GroqProvider(Provider):
    """Groq Provider using instructor for structured output."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Groq provider."""
        if not config.api_key:
            config.api_key = os.getenv("GROQ_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Groq", "GROQ_API_KEY")

        # Create regular Groq client for standard completions
        self.client = groq.Groq(api_key=config.api_key)

        # Create instructor client for structured output
        self.instructor_client = instructor.from_groq(self.client, mode=instructor.Mode.JSON)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Groq."""

        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", "Groq")

        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            # Use instructor for structured output
            instructor_response = self.instructor_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )
            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "groq")

        # Make the API call with regular client
        response: ChatCompletion = self.client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            model=model,
            provider_name="groq",
        )
