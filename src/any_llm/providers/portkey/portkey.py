from typing import Any
from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import Provider, ApiConfig
from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

try:
    from portkey_ai import Portkey
except ImportError as e:
    msg = "Portkey SDK is not installed. Please install it with `pip install any-llm-sdk[portkey]`"
    raise ImportError(msg) from e

class PortkeyProvider(Provider):
    PROVIDER_NAME = "Portkey"
    ENV_API_KEY_NAME = "PORTKEY_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/Portkey-AI/portkey-python-sdk"

    def __init__(self, config: ApiConfig) -> None:
        super().__init__(config)
        self.client = Portkey(api_key=config.api_key)

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
        return self._convert_to_openai_format(response)

    def _convert_to_openai_format(self, response: dict) -> ChatCompletion:
        """Convert Portkey response to OpenAI ChatCompletion format."""
        return ChatCompletion(
            id=response.get("id", ""),
            model=response.get("model", ""),
            object="chat.completion",
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": response.get("content", ""),
                    },
                    "finish_reason": response.get("finish_reason", "stop"),
                }
            ],
            usage={
                "prompt_tokens": response.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": response.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": response.get("usage", {}).get("total_tokens", 0),
            },
        )
