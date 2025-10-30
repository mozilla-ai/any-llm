from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    from xai_sdk import AsyncClient as XaiAsyncClient
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class XaiProvider(AnyLLM):
    API_BASE = "https://api.x.ai/v1"
    ENV_API_KEY_NAME = "XAI_API_KEY"
    PROVIDER_NAME = "xai"
    PROVIDER_DOCUMENTATION_URL = "https://x.ai/"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: XaiAsyncClient

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for xAI API."""
        # xAI does not support providing reasoning effort
        converted_params = params.model_dump(
            exclude_none=True,
            exclude={
                "model_id",
                "messages",
                "stream",
                "response_format",
                "tools",
                "tool_choice",
            },
        )
        if converted_params.get("reasoning_effort") == "auto":
            converted_params.pop("reasoning_effort")
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert xAI response to OpenAI format."""
        return _convert_xai_completion_to_anyllm_response(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert xAI chunk response to OpenAI format."""
        return _convert_xai_chunk_to_anyllm_chunk(response)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for xAI."""
        msg = "xAI does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert xAI embedding response to OpenAI format."""
        msg = "xAI does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert xAI list models response to OpenAI format."""
        return _convert_models_list(response)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = XaiAsyncClient(api_key=api_key, **kwargs)

    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
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

        completion_kwargs = self._convert_completion_params(params, **kwargs)

        chat = self.client.chat.create(
            model=params.model_id,
            messages=xai_messages,
            **completion_kwargs,
        )
        if params.stream:
            if params.response_format:
                err_msg = "Response format is not supported for streaming"
                raise ValueError(err_msg)
            stream_iter: AsyncIterator[tuple[XaiResponse, XaiChunk]] = chat.stream()

            async def _stream() -> AsyncIterator[ChatCompletionChunk]:
                async for _, chunk in stream_iter:
                    yield self._convert_completion_chunk_response(chunk)

            return _stream()

        if params.response_format:
            response, _ = await chat.parse(shape=params.response_format)  # type: ignore[arg-type]
        else:
            response = await chat.sample()

        return self._convert_completion_response(response)

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list_language_models()
        return self._convert_list_models_response(models_list)
