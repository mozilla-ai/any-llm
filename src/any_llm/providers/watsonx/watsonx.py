import os
from typing import Any

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore[attr-defined]
except ImportError:
    msg = "ibm-watsonx-ai is not installed. Please install it with `pip install any-llm-sdk[watsonx]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider
from any_llm.exceptions import UnsupportedParameterError
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from any_llm.providers.watsonx.utils import _convert_response


class WatsonxProvider(Provider):
    """IBM Watsonx Provider using the official IBM Watsonx AI SDK."""

    PROVIDER_NAME = "Watsonx"
    ENV_API_KEY_NAME = "WATSONX_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://www.ibm.com/watsonx"

    SUPPORTS_STREAMING = False
    SUPPORTS_EMBEDDING = False

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Watsonx provider."""
        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Watsonx."""

        model_inference = ModelInference(
            model_id=model,
            credentials=Credentials(
                api_key=self.config.api_key,
                url=self.config.api_base or os.getenv("WATSONX_SERVICE_URL"),
            ),
            project_id=os.getenv("WATSONX_PROJECT_ID"),
        )

        response = model_inference.chat(
            messages=messages,
            params=kwargs,
        )

        return _convert_response(response)
