from collections.abc import AsyncIterator, Sequence
from typing import Any

from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.model import Model

MISSING_PACKAGES_ERROR = None
try:
    from xai_sdk import AsyncClient as XaiAsyncClient
    from xai_sdk import Client as XaiClient
    from xai_sdk.chat import Chunk as XaiChunk
    from xai_sdk.chat import Response as XaiResponse
    from xai_sdk.chat import assistant, required_tool, system, tool_result, user

    from .utils import (
        _convert_models_list,
        _convert_openai_tools_to_xai_tools,
        _convert_xai_chunk_to_anyllm_chunk,
        _convert_xai_completion_to_anyllm_response,
    )

except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class XaiProvider(Provider):
    API_BASE = "https://api.x.ai/v1"
    ENV_API_KEY_NAME = "XAI_API_KEY"
    PROVIDER_NAME = "xai"
    PROVIDER_DOCUMENTATION_URL = "https://x.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Call the XAI Python SDK Chat Completions API and convert to AnyLLM types."""
        client = XaiAsyncClient(api_key=self.config.api_key)

        xai_messages = []
        for message in params.messages:
            if message["role"] == "user":
                xai_messages.append(user(message["content"]))
            elif message["role"] == "assistant":
                args: list[str] = []
                if message.get("tool_calls"):
                    # No idea how to pass tool calls reconstructed in the original protobuf format.
                    args.extend(str(tool_call) for tool_call in message["tool_calls"])
                xai_messages.append(assistant(*args, message["content"]))
            elif message["role"] == "system":
                xai_messages.append(system(message["content"]))
            elif message["role"] == "tool":
                xai_messages.append(tool_result(message["content"]))
        if params.tools is not None:
            kwargs["tools"] = _convert_openai_tools_to_xai_tools(params.tools)

        tool_choice = params.tool_choice
        if isinstance(tool_choice, dict):
            fn = tool_choice.get("function") if tool_choice.get("type") == "function" else None
            name = fn.get("name") if isinstance(fn, dict) else None
            if isinstance(name, str) and name:
                kwargs["tool_choice"] = required_tool(name)
        elif tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        if params.reasoning_effort == "auto":
            params.reasoning_effort = None

        chat = client.chat.create(
            model=params.model_id,
            messages=xai_messages,
            **params.model_dump(
                exclude_none=True,
                exclude={
                    "model_id",
                    "messages",
                    "stream",
                    "response_format",
                    "tools",
                    "tool_choice",
                },
            ),
            **kwargs,
        )
        if params.stream:
            if params.response_format:
                err_msg = "Response format is not supported for streaming"
                raise ValueError(err_msg)
            stream_iter: AsyncIterator[tuple[XaiResponse, XaiChunk]] = chat.stream()

            async def _stream() -> AsyncIterator[ChatCompletionChunk]:
                async for _, chunk in stream_iter:
                    yield _convert_xai_chunk_to_anyllm_chunk(chunk)

            return _stream()

        if params.response_format:
            response, _ = await chat.parse(shape=params.response_format)  # type: ignore[arg-type]
        else:
            response = await chat.sample()

        return _convert_xai_completion_to_anyllm_response(response)

    def list_models(self, **kwargs: Any) -> Sequence[Model]:
        """
        Fetch available models from the /v1/models endpoint.
        """
        client = XaiClient(api_key=self.config.api_key)
        models_list = client.models.list_language_models()
        return _convert_models_list(models_list)
