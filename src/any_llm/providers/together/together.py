from typing import Any, Iterator

try:
    import together
    from together.types import (
        ChatCompletionResponse,
    )
    import instructor
except ImportError:
    msg = "together or instructor is not installed. Please install it with `pip install any-llm-sdk[together]`"
    raise ImportError(msg)


from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.provider import Provider, convert_instructor_response
from any_llm.providers.helpers import create_completion_from_response
from any_llm.providers.together.utils import _create_openai_chunk_from_together_chunk
from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk


class TogetherProvider(Provider):
    PROVIDER_NAME = "Together"
    ENV_API_KEY_NAME = "TOGETHER_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://together.ai/"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = False

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Together provider."""
        pass

    def _stream_completion(
        self,
        client: together.Together,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response: Iterator[TogetherChatCompletionChunk] = client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,
            **kwargs,
        )
        for chunk in response:
            yield _create_openai_chunk_from_together_chunk(chunk)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Make the API call to Together AI with instructor support for structured outputs."""
        if self.config.api_base:
            client = together.Together(api_key=self.config.api_key, base_url=self.config.api_base)
        else:
            client = together.Together(api_key=self.config.api_key)

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            instructor_client = instructor.patch(client, mode=instructor.Mode.JSON)  # type: ignore [call-overload]

            instructor_response = instructor_client.chat.completions.create(
                model=model,
                messages=messages,
                response_model=response_format,
                **kwargs,
            )

            return convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            return self._stream_completion(client, model, messages, **kwargs)

        response: ChatCompletionResponse = client.chat.completions.create(  # type: ignore[assignment]
            model=model,
            messages=messages,
            **kwargs,
        )

        return create_completion_from_response(
            response_data=response.model_dump(),
            model=model,
            provider_name=self.PROVIDER_NAME,
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
