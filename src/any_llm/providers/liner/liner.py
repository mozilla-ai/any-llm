from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import httpx
from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError

from .utils import (
    FORWARDED_EVENTS,
    _convert_event,
    _convert_messages,
    _convert_response,
    _parse_event,
    _raise_if_error_event,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model

DEFAULT_TIMEOUT = 60.0

# Parameters that change what the caller gets back, so Liner cannot silently ignore them.
_REFUSED_PARAMS = ("tools", "tool_choice", "response_format", "parallel_tool_calls")


class LinerProvider(AnyLLM):
    """Liner's answer agents, which search the web and answer with the references they used.

    The model names the agent: ``quick-answer`` for a short answer, ``search`` for a longer one.
    Each reference becomes an entry of ``message.citations`` (``delta.citations`` on a stream),
    the field Exa's answer endpoint uses. The agents take only ``messages``, so sampling
    parameters such as ``temperature``, ``max_tokens`` and ``user`` are dropped, while tools and
    structured output raise ``UnsupportedParameterError``. Keyword arguments such as ``mode``
    are sent in the request body as given.
    """

    PROVIDER_NAME = "liner"
    ENV_API_KEY_NAME = "LINER_API_KEY"
    ENV_API_BASE_NAME = "LINER_API_BASE"
    PROVIDER_DOCUMENTATION_URL = "https://liner.com/developers/docs"
    API_BASE = "https://platform.liner.com/api/v1/agents"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

    TIMEOUT_SUPPORT = "native"

    MISSING_PACKAGES_ERROR = None

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.api_base = (api_base or self.API_BASE).rstrip("/")
        # A client per call, since sync calls run each request on an event loop of its own.
        self.client_kwargs = {"timeout": DEFAULT_TIMEOUT, **kwargs}

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Build the agent request body: the messages, the stream flag, and any extra keyword arguments."""
        for name in _REFUSED_PARAMS:
            if getattr(params, name) is not None:
                raise UnsupportedParameterError(name, "liner")
        return {"messages": _convert_messages(params.messages), "stream": bool(params.stream), **kwargs}

    @staticmethod
    @override
    def _convert_completion_response(response: Any, **kwargs: Any) -> ChatCompletion:
        return _convert_response(response, kwargs.get("model", "unknown"), kwargs.get("response_id", "unknown"))

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        return _convert_event(
            response,
            kwargs.get("model", "unknown"),
            kwargs.get("chunk_id", "unknown"),
            first=kwargs.get("first", False),
        )

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "Liner does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "Liner does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        msg = "Liner does not support listing models"
        raise NotImplementedError(msg)

    def _request(self, model: str, *, stream: bool, timeout: float | None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "url": f"{self.api_base}/{model}",
            "headers": {
                "x-api-key": self.api_key or "",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
            },
        }
        if timeout is not None:
            request["timeout"] = timeout
        return request

    @staticmethod
    def _response_id(response: httpx.Response) -> str:
        return response.headers.get("x-request-id") or f"chatcmpl-{uuid.uuid4().hex}"

    async def _stream_completion_async(
        self, model: str, body: dict[str, Any], request: dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        async with (
            httpx.AsyncClient(**self.client_kwargs) as client,
            client.stream("POST", json=body, **request) as response,
        ):
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            chunk_id = self._response_id(response)
            first = True
            async for line in response.aiter_lines():
                event = _parse_event(line)
                if event is None:
                    continue
                _raise_if_error_event(event)
                if event.get("type") not in FORWARDED_EVENTS:
                    continue
                yield self._convert_completion_chunk_response(event, model=model, chunk_id=chunk_id, first=first)
                first = False

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        timeout = kwargs.pop("timeout", None)
        body = self._convert_completion_params(params, **kwargs)
        request = self._request(params.model_id, stream=bool(params.stream), timeout=timeout)

        if params.stream:
            return self._stream_completion_async(params.model_id, body, request)

        async with httpx.AsyncClient(**self.client_kwargs) as client:
            response = await client.post(json=body, **request)
            response.raise_for_status()
        return self._convert_completion_response(
            response.json(), model=params.model_id, response_id=self._response_id(response)
        )
