import os
from typing import Any, Iterator

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore[attr-defined]
except ImportError as exc:
    msg = "ibm-watsonx-ai is not installed. Please install it with `pip install any-llm-sdk[watsonx]`"
    raise ImportError(msg) from exc

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk
from any_llm.provider import Provider
from any_llm.providers.watsonx.utils import _convert_response, _convert_streaming_chunk


class WatsonxProvider(Provider):
    """IBM Watsonx Provider using the official IBM Watsonx AI SDK."""

    PROVIDER_NAME = "Watsonx"
    ENV_API_KEY_NAME = "WATSONX_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://www.ibm.com/watsonx"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = False

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Watsonx provider."""

    def _stream_completion(
        self,
        model_inference: ModelInference,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response_stream = model_inference.chat_stream(
            messages=messages,
            params=kwargs,
        )
        for chunk in response_stream:
            yield _convert_streaming_chunk(chunk)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Watsonx."""

        model_inference = ModelInference(
            model_id=model,
            credentials=Credentials(
                api_key=self.config.api_key,
                url=self.config.api_base or os.getenv("WATSONX_SERVICE_URL"),
            ),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
        )

        if kwargs.get("stream", False):
            return self._stream_completion(model_inference, messages, **kwargs)

        response = model_inference.chat(
            messages=messages,
            params=kwargs,
        )

        return _convert_response(response)
