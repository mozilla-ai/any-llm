from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import ProviderError, UnsupportedParameterError
from any_llm.utils.embedding import apply_encoding_format, pop_encoding_format
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    import cohere
    from cohere import V2ChatResponse

    from .utils import (
        _convert_cohere_embedding_response,
        _convert_cohere_rerank_response,
        _convert_models_list,
        _convert_response,
        _create_openai_chunk_from_cohere_chunk,
        _patch_messages,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model
    from any_llm.types.rerank import RerankResponse


REASONING_EFFORT_TO_THINKING_BUDGETS = {
    "minimal": 256,
    "low": 1024,
    "medium": 8192,
    "high": 24576,
    "xhigh": 32768,
    "max": 32768,
}


class CohereProvider(AnyLLM):
    """Cohere Provider using the new response conversion utilities."""

    PROVIDER_NAME = "cohere"
    ENV_API_KEY_NAME = "COHERE_API_KEY"
    ENV_API_BASE_NAME = "COHERE_BASE_URL"
    PROVIDER_DOCUMENTATION_URL = "https://cohere.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_RERANK = True

    # The Cohere SDK carries a per-request timeout inside `request_options`, not as a top-level
    # keyword, so it is translated in _convert_completion_params.
    TIMEOUT_SUPPORT = "mapped"
    MAX_RETRIES_SUPPORT = "native"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: cohere.AsyncClientV2

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Cohere API."""
        converted_params = params.model_dump(
            exclude_none=True,
            exclude={"model_id", "messages", "response_format", "stream", "stream_options", "reasoning_effort"},
        )
        if params.reasoning_effort != "auto":
            if params.reasoning_effort is None or params.reasoning_effort == "none":
                converted_params["thinking"] = {"type": "disabled"}
            else:
                converted_params["thinking"] = {
                    "type": "enabled",
                    "token_budget": REASONING_EFFORT_TO_THINKING_BUDGETS[params.reasoning_effort],
                }
        converted_params.update(kwargs)

        # Cohere's SDK carries a per-request timeout inside `request_options` rather than as a
        # top-level keyword, so the seconds-based any-llm parameter has to be folded in there.
        if (timeout := converted_params.pop("timeout", None)) is not None:
            request_options = dict(converted_params.get("request_options") or {})
            request_options.setdefault("timeout", timeout)
            converted_params["request_options"] = request_options

        return converted_params

    @staticmethod
    @override
    def _convert_completion_response(response: V2ChatResponse, **kwargs: Any) -> ChatCompletion:
        """Convert Cohere response to OpenAI format."""
        # We need the model parameter for conversion
        model = kwargs.get("model", "unknown")
        return _convert_response(response, model)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Cohere chunk response to OpenAI format."""
        model = kwargs.get("model", "unknown")
        return _create_openai_chunk_from_cohere_chunk(response, model)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Cohere.

        Cohere requires ``input_type`` (one of ``search_document``,
        ``search_query``, ``classification``, ``clustering``, ``image``).
        Callers may pass it via **kwargs; the default is ``search_document``.
        """
        if isinstance(params, str):
            params = [params]
        converted: dict[str, Any] = {"texts": params}
        converted["input_type"] = kwargs.pop("input_type", "search_document")
        converted["embedding_types"] = kwargs.pop("embedding_types", ["float"])
        converted.update(kwargs)
        return converted

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Cohere EmbedByTypeResponse to OpenAI CreateEmbeddingResponse."""
        model = response.get("model", "cohere")
        return _convert_cohere_embedding_response(model, response["result"])

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        encoding_format = pop_encoding_format(kwargs, self.PROVIDER_NAME)
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)
        result = await self.client.embed(model=model, **embedding_kwargs)
        response_data = {"model": model, "result": result}
        return apply_encoding_format(self._convert_embedding_response(response_data), encoding_format)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Cohere list models response to OpenAI format."""
        return _convert_models_list(response)

    @staticmethod
    @override
    def _convert_rerank_params(model: str, query: str, documents: list[str], **kwargs: Any) -> dict[str, Any]:
        """Convert rerank parameters for Cohere API."""
        params: dict[str, Any] = {
            "query": query,
            "documents": documents,
        }
        if "top_n" in kwargs and kwargs["top_n"] is not None:
            params["top_n"] = kwargs["top_n"]
        if "max_tokens_per_doc" in kwargs and kwargs["max_tokens_per_doc"] is not None:
            params["max_tokens_per_doc"] = kwargs["max_tokens_per_doc"]
        for key in ("return_documents",):
            if key in kwargs:
                params[key] = kwargs[key]
        return params

    @staticmethod
    @override
    def _convert_rerank_response(response: Any) -> RerankResponse:
        """Convert Cohere rerank response to normalized RerankResponse."""
        return _convert_cohere_rerank_response(response)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = cohere.AsyncClientV2(api_key=api_key, **kwargs)

    async def _stream_completion_async(
        self, model: str, messages: list[dict[str, Any]], *, thinking_from_reasoning_effort: bool, **kwargs: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        cohere_stream = self.client.chat_stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        try:
            async for chunk in cohere_stream:
                yield self._convert_completion_chunk_response(chunk, model=model)
        except cohere.BadRequestError as e:
            self._raise_if_thinking_unsupported(e, thinking_from_reasoning_effort)
            raise

    def _raise_if_thinking_unsupported(
        self, error: cohere.BadRequestError, thinking_from_reasoning_effort: bool
    ) -> None:
        """Report Cohere's rejection of `thinking` as the `reasoning_effort` the caller sent.

        Only some Cohere models accept `thinking`, and Cohere's error names the translated field,
        which the caller never sent.
        """
        body = error.body
        message = body.get("message", "") if isinstance(body, dict) else str(body)
        if thinking_from_reasoning_effort and "`thinking` parameter is not supported" in str(message):
            msg = "reasoning_effort"
            raise UnsupportedParameterError(
                msg, self.PROVIDER_NAME, "This Cohere model does not support reasoning."
            ) from error

    @staticmethod
    def _preprocess_response_format(response_format: type | dict[str, Any]) -> dict[str, Any]:
        """Convert a response_format to the shape Cohere's v2 chat endpoint accepts.

        Cohere takes `{"type": "json_object", "schema": ...}` (or no schema for plain JSON mode)
        and `{"type": "text"}`. The OpenAI `{"type": "json_schema", "json_schema": {...}}` shape
        is unwrapped into that, dropping `name`, `description` and `strict`: Cohere has no
        equivalent for them, and it validates whatever sits under `json_schema` as the schema
        itself, so passing the OpenAI shape through is a 400. Other dicts pass through.

        See https://docs.cohere.com/docs/structured-outputs
        """
        if is_structured_output_type(response_format):
            return {"type": "json_object", "schema": get_json_schema(response_format)}
        if isinstance(response_format, dict):
            if response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
                if not isinstance(schema, dict):
                    msg = "response_format of type 'json_schema' must carry a schema under json_schema.schema"
                    raise ValueError(msg)
                return {"type": "json_object", "schema": schema}
            return response_format
        msg = f"Unsupported response_format: {response_format}"
        raise ValueError(msg)

    @classmethod
    def _convert_tools(cls, tools: list[dict[str, Any] | Any]) -> tuple[list[dict[str, Any] | Any], bool]:
        """Strip OpenAI's per-function `strict` flag and report whether the function tools set it.

        Cohere rejects `strict` inside a function definition and instead takes one request-level
        `strict_tools` flag that applies to every tool. A mix of strict and non-strict function
        tools therefore has no faithful mapping: turning `strict_tools` off would drop the
        guarantee the strict tools asked for, and turning it on would impose strict mode (which
        rejects tools with no required parameter) on tools that did not ask for it.
        """
        converted: list[dict[str, Any] | Any] = []
        strict_flags: list[bool] = []
        for tool in tools:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                function = dict(tool["function"])
                strict_flags.append(function.pop("strict", None) is True)
                converted.append({**tool, "function": function})
            else:
                converted.append(tool)
        if any(strict_flags) and not all(strict_flags):
            msg = "strict"
            raise UnsupportedParameterError(
                msg,
                cls.PROVIDER_NAME,
                "Cohere applies strict mode to all tools of a request, so function tools must all be strict or all"
                " non-strict.",
            )
        return converted, any(strict_flags)

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.response_format is not None:
            kwargs["response_format"] = self._preprocess_response_format(params.response_format)
        if params.tools:
            kwargs["tools"], strict_tools = self._convert_tools(params.tools)
            if strict_tools:
                kwargs.setdefault("strict_tools", True)
        # Cohere's tool_choice is only REQUIRED or NONE; it cannot name a tool.
        if isinstance(params.tool_choice, dict):
            msg = "tool_choice"
            raise UnsupportedParameterError(
                msg, self.PROVIDER_NAME, "Cohere accepts only 'required' or 'none', not a named tool."
            )
        if params.parallel_tool_calls is not None:
            msg = "parallel_tool_calls"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        completion_kwargs = self._convert_completion_params(params, **kwargs)
        patched_messages = _patch_messages(params.messages)
        thinking_from_reasoning_effort = params.reasoning_effort != "auto" and "thinking" not in kwargs

        if params.stream:
            return self._stream_completion_async(
                params.model_id,
                patched_messages,
                thinking_from_reasoning_effort=thinking_from_reasoning_effort,
                **completion_kwargs,
            )

        try:
            response = await self.client.chat(
                model=params.model_id,
                messages=patched_messages,  # type: ignore[arg-type]
                **completion_kwargs,
            )
        except cohere.BadRequestError as e:
            self._raise_if_thinking_unsupported(e, thinking_from_reasoning_effort)
            raise

        return self._convert_completion_response(response, model=params.model_id)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        # Cohere pages the catalog (20 models per page by default), so follow next_page_token to the end.
        models: list[Model] = []
        requested_tokens = {kwargs.get("page_token")}
        while True:
            page = await self.client.models.list(**kwargs)
            models.extend(self._convert_list_models_response(page))
            if not page.next_page_token:
                return models
            if page.next_page_token in requested_tokens:
                msg = f"Cohere returned the already requested page token {page.next_page_token!r} as the next page."
                raise ProviderError(msg, provider_name=self.PROVIDER_NAME)
            requested_tokens.add(page.next_page_token)
            kwargs["page_token"] = page.next_page_token

    @override
    async def _arerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> RerankResponse:
        rerank_kwargs = self._convert_rerank_params(model, query, documents, **kwargs)
        response = await self.client.rerank(
            model=model,
            **rerank_kwargs,
        )
        return self._convert_rerank_response(response)
