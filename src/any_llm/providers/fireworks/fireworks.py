from typing import Any, Iterator

try:
    from fireworks import LLM
except ImportError:
    msg = "fireworks-ai is not installed. Please install it with `pip install any-llm-sdk[fireworks]`"
    raise ImportError(msg)

from pydantic import BaseModel
from any_llm.types.completion import ChatCompletionChunk, ChatCompletion
from any_llm.provider import Provider
from any_llm.providers.helpers import create_completion_from_response
from any_llm.providers.fireworks.utils import _create_openai_chunk_from_fireworks_chunk


class FireworksProvider(Provider):
    PROVIDER_NAME = "Fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"

    SUPPORTS_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
    SUPPORTS_EMBEDDING = False

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Fireworks provider."""
        pass

    def _stream_completion(
        self,
        llm: LLM,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        response_generator = llm.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        for chunk in response_generator:
            yield _create_openai_chunk_from_fireworks_chunk(chunk)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        llm = LLM(
            model=model,
            deployment_type="auto",
            api_key=self.config.api_key,
        )

        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
                }
            else:
                kwargs["response_format"] = response_format

        if kwargs.get("stream", False):
            return self._stream_completion(llm, messages, **kwargs)

        response = llm.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        return create_completion_from_response(
            response_data=response.model_dump(),
            provider_name="Fireworks",
            model=model,
        )
