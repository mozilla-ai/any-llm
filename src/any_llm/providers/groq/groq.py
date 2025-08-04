from typing import Any

try:
    import groq
    import instructor
except ImportError:
    msg = "groq or instructor is not installed. Please install it with `pip install any-llm-sdk[groq]`"
    raise ImportError(msg)


from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, convert_instructor_response
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.helpers import create_completion_from_response


class GroqProvider(Provider):
    """Groq Provider using instructor for structured output."""

    PROVIDER_NAME = "Groq"
    ENV_API_KEY_NAME = "GROQ_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://groq.com/api"

    SUPPORTS_STREAMING = False
    SUPPORTS_EMBEDDING = False

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Groq provider."""
        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Groq."""
        # Create regular Groq client for standard completions
        client = groq.Groq(api_key=self.config.api_key)

        # Create instructor client for structured output

        # Handle response_format for structured output
        if "response_format" in kwargs:
            instructor_client = instructor.from_groq(client, mode=instructor.Mode.JSON)
            response_format = kwargs.pop("response_format")
            # Use instructor for structured output
            instructor_response = instructor_client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )
            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        # Make the API call with regular client
        response: ChatCompletion = client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            model=model,
            provider_name=self.PROVIDER_NAME,
        )
