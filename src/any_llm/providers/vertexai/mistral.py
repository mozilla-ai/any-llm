"""Mistral models on Vertex AI.

Vertex serves Mistral models from the `mistralai` publisher, not the `google` one that the genai SDK
targets, through `:rawPredict` and `:streamRawPredict`. Both take and return Mistral's chat completion
format, which is close enough to OpenAI's that the OpenAI SDK can send the request and parse the JSON
and SSE replies. See https://docs.cloud.google.com/vertex-ai/generative-ai/docs/partner-models/mistral
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm.providers.openai.base import BaseOpenAIProvider, OpenAIChunkStream
from any_llm.utils.reasoning import strip_extra_content
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai._types import RequestOptions

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams

MISTRAL_MODEL_PREFIXES = ("mistral", "codestral")

# Mistral accepts a reasoning_effort of only "high" or "none"; see the mistral provider.
_MISTRAL_REASONING_EFFORT = "high"

# The bearer token is set per request, since Vertex access tokens expire. The OpenAI SDK requires a
# client-level key, so it gets this placeholder, which the per-request header overrides.
_PLACEHOLDER_API_KEY = "vertex-access-token"


def is_mistral_model(model_id: str) -> bool:
    """Return whether Vertex serves `model_id` from the `mistralai` publisher."""
    return model_id.startswith(MISTRAL_MODEL_PREFIXES)


def mistral_base_url(project: str, location: str) -> str:
    """Return the URL under which Vertex serves the `mistralai` publisher's models."""
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/publishers/mistralai/models"


def create_mistral_client(project: str, location: str, timeout: float | None = None) -> AsyncOpenAI:
    """Create an OpenAI SDK client aimed at the `mistralai` publisher on Vertex."""
    client_kwargs: dict[str, Any] = {"timeout": timeout} if timeout is not None else {}
    return AsyncOpenAI(api_key=_PLACEHOLDER_API_KEY, base_url=mistral_base_url(project, location), **client_kwargs)


def convert_mistral_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
    """Convert CompletionParams to a Mistral chat completion request body."""
    body = params.model_dump(
        exclude_none=True,
        exclude={
            "model_id",
            "messages",
            "response_format",
            "reasoning_effort",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "user",
        },
    )
    # Vertex documents that the body names the model without its `@version` suffix.
    body["model"] = params.model_id.split("@", 1)[0]
    body["messages"] = strip_extra_content(params.messages)
    body["stream"] = bool(params.stream)

    # Mistral names the output limit max_tokens and does not accept max_completion_tokens.
    if params.max_tokens is None and params.max_completion_tokens is not None:
        body["max_tokens"] = params.max_completion_tokens

    if is_structured_output_type(params.response_format):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": params.response_format.__name__,
                "schema": get_json_schema(params.response_format),
            },
        }
    elif isinstance(params.response_format, dict):
        body["response_format"] = params.response_format

    if params.reasoning_effort is not None and params.reasoning_effort not in ("auto", "none"):
        body["reasoning_effort"] = _MISTRAL_REASONING_EFFORT

    body.update(kwargs)
    return body


async def acompletion_mistral(
    client: AsyncOpenAI, access_token: str, params: CompletionParams, **kwargs: Any
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """Call a Mistral model through Vertex's `rawPredict` or `streamRawPredict` endpoint."""
    timeout = kwargs.pop("timeout", None)
    body = convert_mistral_params(params, **kwargs)
    options: RequestOptions = {"headers": {"Authorization": f"Bearer {access_token}"}}
    if timeout is not None:
        options["timeout"] = timeout
    # The leading slash keeps httpx from reading the `<model>:` prefix of the path as a URL scheme.
    method = "streamRawPredict" if params.stream else "rawPredict"
    path = f"/{params.model_id}:{method}"

    if params.stream:
        stream = await client.post(
            path,
            cast_to=OpenAIChatCompletion,
            body=body,
            options=options,
            stream=True,
            stream_cls=AsyncStream[OpenAIChatCompletionChunk],
        )
        return OpenAIChunkStream(stream, BaseOpenAIProvider._convert_completion_chunk_response)

    response = await client.post(
        path,
        cast_to=OpenAIChatCompletion,
        body=body,
        options=options,
    )
    return BaseOpenAIProvider._convert_completion_response(response)
