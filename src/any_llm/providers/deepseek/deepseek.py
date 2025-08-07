from typing import Any, Iterator

from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletionChunk, ChatCompletion
from any_llm.providers.deepseek.utils import _convert_pydantic_to_deepseek_json


class DeepseekProvider(BaseOpenAIProvider):
    API_BASE = "https://api.deepseek.com"
    ENV_API_KEY_NAME = "DEEPSEEK_API_KEY"
    PROVIDER_NAME = "DeepSeek"
    PROVIDER_DOCUMENTATION_URL = "https://platform.deepseek.com/"

    SUPPORTS_EMBEDDING = False  # DeepSeek doesn't host an embedding model

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        # Handle Pydantic model conversion for DeepSeek
        if "response_format" in kwargs:
            response_format = kwargs["response_format"]
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                modified_messages = _convert_pydantic_to_deepseek_json(response_format, messages)
                kwargs["response_format"] = {"type": "json_object"}
                messages = modified_messages

        return super()._make_api_call(model, messages, **kwargs)
