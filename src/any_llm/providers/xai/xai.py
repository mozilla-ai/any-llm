from collections.abc import Iterator
from typing import Any

from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

try:
    from xai_sdk import Client as XaiClient
    from xai_sdk.chat import Chunk as XaiChunk
    from xai_sdk.chat import Response as XaiResponse
    from xai_sdk.chat import assistant, system, user

    from any_llm.providers.xai.utils import (
        _convert_xai_chunk_to_anyllm_chunk,
        _convert_xai_completion_to_anyllm_response,
    )
except ImportError as exc:
    msg = "xai is not installed. Please install it with `pip install any-llm-sdk[xai]`"
    raise ImportError(msg) from exc


class XaiProvider(Provider):
    API_BASE = "https://api.x.ai/v1"
    ENV_API_KEY_NAME = "XAI_API_KEY"
    PROVIDER_NAME = "xAI"
    PROVIDER_DOCUMENTATION_URL = "https://x.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    def completion(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Call the XAI Python SDK Chat Completions API and convert to AnyLLM types."""
        client = XaiClient(api_key=self.config.api_key)

        xai_messages = []
        for message in messages:
            if message["role"] == "user":
                xai_messages.append(user(message["content"]))
            elif message["role"] == "assistant":
                xai_messages.append(assistant(message["content"]))
            elif message["role"] == "system":
                xai_messages.append(system(message["content"]))
        response_format = kwargs.pop("response_format", None)
        chat = client.chat.create(
            model=model,
            messages=xai_messages,
            **kwargs,
        )
        if kwargs.get("stream", False):
            if response_format:
                err_msg = "Response format is not supported for streaming"
                raise ValueError(err_msg)
            stream_iter: Iterator[tuple[XaiResponse, XaiChunk]] = chat.stream()
            return (_convert_xai_chunk_to_anyllm_chunk(chunk) for _, chunk in stream_iter)

        if response_format:
            response, _ = chat.parse(shape=response_format)
        else:
            response = chat.sample()

        return _convert_xai_completion_to_anyllm_response(response)
