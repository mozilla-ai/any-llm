from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    from lmstudio import AsyncClient, Chat

    from .utils import (
        convert_models_list,
        create_chunk_from_fragment,
        create_completion_from_prediction,
        create_embedding_response,
        create_final_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model


class LmstudioProvider(AnyLLM):
    """LM Studio provider backed by the native `lmstudio-python` SDK.

    Unlike the OpenAI-compatible REST endpoint, the native SDK speaks LM Studio's
    websocket protocol. Its async client requires structured concurrency
    (`async with AsyncClient() as client: ...`), so a fresh client is opened for
    the duration of each request rather than being reused across calls.

    Read more here - https://lmstudio.ai/docs/python
    """

    PROVIDER_NAME = "lmstudio"
    PROVIDER_DOCUMENTATION_URL = "https://lmstudio.ai/docs/python"
    ENV_API_KEY_NAME = "LM_STUDIO_API_KEY"
    ENV_API_BASE_NAME = "LM_STUDIO_API_BASE"
    API_BASE = "localhost:1234"

    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_MODERATION = False
    SUPPORTS_BATCH = False
    SUPPORTS_RERANK = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    api_host: str

    @staticmethod
    def _normalize_api_host(api_base: str | None) -> str | None:
        """Reduce an OpenAI-style base URL to the `host:port` the SDK expects.

        Accepts values such as `http://localhost:1234/v1` (the legacy default) and
        returns `localhost:1234`. Returns None when no usable host is provided.
        """
        if not api_base:
            return None
        host = api_base.strip()
        for scheme in ("http://", "https://", "ws://", "wss://"):
            if host.startswith(scheme):
                host = host[len(scheme) :]
                break
        host = host.rstrip("/")
        host = host.removesuffix("/v1")
        return host.rstrip("/") or None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # LM Studio runs locally and does not require an API key.
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_host = self._normalize_api_host(api_base) or self.API_BASE

    def _build_chat(self, messages: list[dict[str, Any]]) -> Chat:
        """Convert OpenAI-style messages into an LM Studio `Chat` history."""
        chat = Chat()
        for message in messages:
            role = message.get("role")
            content = self._extract_text_content(message.get("content"))
            if role == "system":
                chat.add_system_prompt(content)
            elif role == "assistant":
                chat.add_assistant_response(content)
            else:
                # Map user (and any unsupported roles such as tool) to a user turn
                # so prior context is preserved even though tool calling is unsupported.
                chat.add_user_message(content)
        return chat

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Flatten message content into plain text (images/files are not supported)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
            ]
            return " ".join(text_parts)
        return str(content)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Build an LM Studio prediction config dict from the shared CompletionParams."""
        config: dict[str, Any] = {}
        if params.temperature is not None:
            config["temperature"] = params.temperature
        max_tokens = params.max_completion_tokens or params.max_tokens
        if max_tokens is not None:
            config["maxTokens"] = max_tokens
        if params.top_p is not None:
            config["topPSampling"] = params.top_p
        if params.stop is not None:
            config["stopStrings"] = [params.stop] if isinstance(params.stop, str) else params.stop
        config.update(kwargs)
        return config

    @staticmethod
    def _resolve_response_format(params: CompletionParams) -> dict[str, Any] | None:
        """Translate response_format into a JSON schema the SDK can enforce."""
        if params.response_format is None:
            return None
        if is_structured_output_type(params.response_format):
            return get_json_schema(params.response_format)
        if isinstance(params.response_format, dict):
            return params.response_format
        return None

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert an LM Studio PredictionResult to an OpenAI-compatible ChatCompletion."""
        return create_completion_from_prediction(response, response.model_info.model_key)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert an LM Studio prediction fragment to an OpenAI-compatible chunk."""
        model = kwargs.get("model", "unknown")
        response_id = kwargs.get("response_id", f"chatcmpl-{uuid.uuid4()}")
        return create_chunk_from_fragment(response, model, response_id)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        converted_params = {"input": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert normalized embedding output ({"model", "vectors"}) to OpenAI format."""
        return create_embedding_response(response["model"], response["vectors"])

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        return convert_models_list(response)

    async def _stream_completion(
        self,
        params: CompletionParams,
        config: dict[str, Any],
        response_format: dict[str, Any] | None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        chat = self._build_chat(params.messages)
        response_id = f"chatcmpl-{uuid.uuid4()}"

        async with AsyncClient(api_host=self.api_host) as client:
            model = await client.llm.model(params.model_id)
            prediction_stream = await model.respond_stream(
                chat,
                config=cast("Any", config or None),
                response_format=response_format,
            )
            async for fragment in prediction_stream:
                yield create_chunk_from_fragment(fragment, params.model_id, response_id)
            result = prediction_stream.result()
            yield create_final_chunk(result, params.model_id, response_id)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.tools:
            msg = (
                "The LM Studio provider uses the native lmstudio-python SDK, which does not support "
                "OpenAI-style tool calling (returning tool calls without executing them). "
                "Tool use is only available via the SDK's agentic `.act()` API, which is not exposed here."
            )
            raise NotImplementedError(msg)

        config = self._convert_completion_params(params, **kwargs)
        response_format = self._resolve_response_format(params)

        if params.stream:
            return self._stream_completion(params, config, response_format)

        chat = self._build_chat(params.messages)
        async with AsyncClient(api_host=self.api_host) as client:
            model = await client.llm.model(params.model_id)
            result = await model.respond(
                chat,
                config=cast("Any", config or None),
                response_format=response_format,
            )
        return create_completion_from_prediction(result, params.model_id)

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        async with AsyncClient(api_host=self.api_host) as client:
            embedding_model = await client.embedding.model(model)
            result = await embedding_model.embed(inputs)

        if isinstance(inputs, str):
            vectors = [list(cast("Sequence[float]", result))]
        else:
            vectors = [list(vector) for vector in cast("Sequence[Sequence[float]]", result)]
        return create_embedding_response(model, vectors)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        async with AsyncClient(api_host=self.api_host) as client:
            downloaded = await client.list_downloaded_models()
        return convert_models_list(downloaded)
