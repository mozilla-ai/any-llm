from typing import Any

try:
    from fireworks import LLM
except ImportError:
    msg = "fireworks-ai is not installed. Please install it with `pip install any-llm-sdk[fireworks]`"
    raise ImportError(msg)

from pydantic import BaseModel
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.helpers import create_completion_from_response


class FireworksProvider(Provider):
    PROVIDER_NAME = "Fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"

    SUPPORTS_STREAMING = False
    SUPPORTS_EMBEDDING = False

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Fireworks provider."""
        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        llm = LLM(
            model=model,
            deployment_type="serverless",
            api_key=self.config.api_key,
        )

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to Fireworks JSON schema format
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
                }
            else:
                # response_format is already a dict, pass it through
                kwargs["response_format"] = response_format

        # Make the API call
        response = llm.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            provider_name="Fireworks",
            model=model,
        )
