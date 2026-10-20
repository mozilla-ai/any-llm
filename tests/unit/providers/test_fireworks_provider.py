import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from openai import DEFAULT_TIMEOUT
from pydantic import BaseModel

from any_llm.exceptions import ModelNotFoundError, UnsupportedParameterError
from any_llm.providers.fireworks import FireworksProvider
from any_llm.providers.fireworks.fireworks import _sdk_base_url
from any_llm.providers.fireworks.utils import convert_fireworks_rerank_response, extract_reasoning_from_response
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    ParsedChatCompletion,
    Reasoning,
)
from any_llm.types.responses import Response


def test_extract_reasoning_from_response_with_think_tags() -> None:
    """Test that <think> content is correctly extracted into reasoning field."""
    # Create a mock Response with <think> tags in content
    mock_content = Mock()
    mock_content.text = "<think>This is my reasoning process</think>This is the actual response"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is not None
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "This is my reasoning process"
    assert mock_content.text == "This is the actual response"


def test_extract_reasoning_from_response_preserves_content_before_the_tag() -> None:
    """Content emitted before the opening tag is part of the answer, not reasoning."""
    mock_content = Mock()
    mock_content.text = "Preamble text. <think>my reasoning</think>The answer"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is not None
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "my reasoning"
    assert mock_content.text == "Preamble text. The answer"


def test_extract_reasoning_from_response_handles_multiple_tagged_blocks() -> None:
    """Every tagged block is collected, and no tag is left behind in the content."""
    mock_content = Mock()
    mock_content.text = "<think>first</think>middle<think>second</think>end"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is not None
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "first\nsecond"
    assert mock_content.text == "middleend"
    assert "<think>" not in mock_content.text


def test_extract_reasoning_from_response_keeps_mixed_tags_in_textual_order() -> None:
    mock_content = Mock()
    mock_content.text = "<think>first</think>middle<thinking>second</thinking>end"
    mock_output_item = Mock()
    mock_output_item.content = [mock_content]
    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "first\nsecond"
    assert mock_content.text == "middleend"


def test_extract_reasoning_from_response_handles_alternate_tag_names() -> None:
    """Reasoning tags other than <think> are recognised, matching the shared normalizer."""
    mock_content = Mock()
    mock_content.text = "<thinking>alt tag reasoning</thinking>The answer"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is not None
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "alt tag reasoning"
    assert mock_content.text == "The answer"


def test_extract_reasoning_from_response_leaves_existing_reasoning_alone() -> None:
    """A response that already carries reasoning is returned untouched."""
    mock_content = Mock()
    mock_content.text = "<think>ignored</think>The answer"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = Reasoning(content="already set")

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is not None
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "already set"
    assert mock_content.text == "<think>ignored</think>The answer"


def test_extract_reasoning_from_response_without_think_tags() -> None:
    """Test that responses without <think> tags are returned unchanged."""
    mock_content = Mock()
    mock_content.text = "This is just a regular response"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is None
    assert mock_content.text == "This is just a regular response"


def test_extract_reasoning_from_response_empty_reasoning() -> None:
    """Test that empty reasoning content is handled correctly."""
    mock_content = Mock()
    mock_content.text = "<think></think>This is the actual response"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is None
    assert mock_content.text == "This is the actual response"


def test_extract_reasoning_from_response_empty_output() -> None:
    """Test that responses with empty output are handled gracefully."""
    mock_response = Mock(spec=Response)
    mock_response.output = []
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is None
    assert result == mock_response


def test_extract_reasoning_from_response_leaves_a_trailing_tool_call_alone() -> None:
    """A response ending in a function_call has no content to scan, so it is returned unchanged."""
    response = Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 0,
            "model": "accounts/fireworks/models/kimi-k3",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "get_number",
                    "arguments": "{}",
                    "status": "completed",
                }
            ],
        }
    )

    result = extract_reasoning_from_response(response)

    assert result.reasoning is None
    assert result.output[0].type == "function_call"


RERANK_MODEL = "accounts/fireworks/models/qwen3-reranker-8b"


def _fireworks_with_transport(handler: httpx.MockTransport, **kwargs: Any) -> FireworksProvider:
    """Build a provider whose OpenAI client sends every request to `handler`."""
    return FireworksProvider(
        api_key="test-key",
        api_base="https://fireworks.test/inference/v1",
        http_client=httpx.AsyncClient(transport=handler),
        **kwargs,
    )


def test_fireworks_supports_rerank() -> None:
    """Fireworks advertises rerank support."""
    assert FireworksProvider.SUPPORTS_RERANK is True


def test_convert_rerank_params_defaults() -> None:
    """Only model, query, documents and return_documents=False are sent by default."""
    params = FireworksProvider._convert_rerank_params(RERANK_MODEL, "q", ["a", "b"])
    assert params == {"model": RERANK_MODEL, "query": "q", "documents": ["a", "b"], "return_documents": False}


def test_convert_rerank_params_passes_top_n_task_and_return_documents() -> None:
    """top_n, task and an explicit return_documents pass through."""
    params = FireworksProvider._convert_rerank_params(
        RERANK_MODEL, "q", ["a"], top_n=1, task="Find the answer", return_documents=True
    )
    assert params["top_n"] == 1
    assert params["task"] == "Find the answer"
    assert params["return_documents"] is True


def test_convert_rerank_params_drops_none_top_n() -> None:
    """A None top_n is not sent."""
    params = FireworksProvider._convert_rerank_params(RERANK_MODEL, "q", ["a"], top_n=None)
    assert "top_n" not in params


def test_convert_rerank_params_rejects_max_tokens_per_doc() -> None:
    """max_tokens_per_doc has no Fireworks equivalent."""
    with pytest.raises(UnsupportedParameterError):
        FireworksProvider._convert_rerank_params(RERANK_MODEL, "q", ["a"], max_tokens_per_doc=16)


def test_convert_rerank_params_ignores_none_max_tokens_per_doc() -> None:
    """A None max_tokens_per_doc is accepted and not sent."""
    params = FireworksProvider._convert_rerank_params(RERANK_MODEL, "q", ["a"], max_tokens_per_doc=None)
    assert "max_tokens_per_doc" not in params


def test_convert_fireworks_rerank_response_sorts_and_maps_usage() -> None:
    """Results are sorted by score, the ID stays None and usage is mapped."""
    result = convert_fireworks_rerank_response(
        {
            "object": "list",
            "model": RERANK_MODEL,
            "data": [
                {"index": 0, "relevance_score": 0.1, "document": "a"},
                {"index": 1, "relevance_score": 0.9},
            ],
            "usage": {"prompt_tokens": 12, "total_tokens": 12},
        }
    )
    assert result.id is None
    assert result.model == RERANK_MODEL
    assert [(r.index, r.relevance_score) for r in result.results] == [(1, 0.9), (0, 0.1)]
    assert result.usage is not None
    assert result.usage.total_tokens == 12


def test_convert_fireworks_rerank_response_without_usage() -> None:
    """A response without usage converts with usage=None."""
    result = convert_fireworks_rerank_response({"model": RERANK_MODEL, "data": []})
    assert result.results == []
    assert result.usage is None


@pytest.mark.asyncio
async def test_arerank_posts_to_fireworks_rerank_endpoint() -> None:
    """arerank sends the converted body to <api_base>/rerank with the bearer key."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": RERANK_MODEL,
                "data": [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.2}],
                "usage": {"prompt_tokens": 20, "total_tokens": 20},
            },
        )

    provider = _fireworks_with_transport(httpx.MockTransport(respond))
    result = await provider.arerank(RERANK_MODEL, "capital of France", ["Berlin", "Paris"], top_n=2)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://fireworks.test/inference/v1/rerank"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": RERANK_MODEL,
        "query": "capital of France",
        "documents": ["Berlin", "Paris"],
        "return_documents": False,
        "top_n": 2,
    }
    assert result.model == RERANK_MODEL
    assert [r.index for r in result.results] == [1, 0]
    assert result.usage is not None
    assert result.usage.total_tokens == 20


@pytest.mark.asyncio
async def test_arerank_maps_provider_errors() -> None:
    """A Fireworks 404 surfaces as ModelNotFoundError with unified exceptions."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "Model not found"}})

    provider = _fireworks_with_transport(httpx.MockTransport(respond), unified_exceptions=True)
    with pytest.raises(ModelNotFoundError):
        await provider.arerank("accounts/fireworks/models/nope", "q", ["a"])


@pytest.mark.asyncio
@pytest.mark.parametrize("request_timeout", [None, 3.0])
async def test_arerank_forwards_per_request_timeout(request_timeout: float | None) -> None:
    """A per-request timeout reaches the HTTP request and stays out of the body."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"object": "list", "model": RERANK_MODEL, "data": []})

    provider = _fireworks_with_transport(httpx.MockTransport(respond))
    kwargs: dict[str, Any] = {} if request_timeout is None else {"timeout": request_timeout}
    await provider.arerank(RERANK_MODEL, "q", ["a"], **kwargs)

    assert "timeout" not in json.loads(requests[0].content)
    read_timeout = requests[0].extensions["timeout"]["read"]
    assert read_timeout == (request_timeout if request_timeout is not None else DEFAULT_TIMEOUT.read)


COMPLETION_MODEL = "accounts/fireworks/models/gpt-oss-120b"
USER_MESSAGES = [{"role": "user", "content": "Capital of France?"}]


class _City(BaseModel):
    name: str


def _completion_body(**message: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": COMPLETION_MODEL,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Paris", **message}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"


def _chunk(delta: dict[str, Any], finish_reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": COMPLETION_MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("https://api.fireworks.ai/inference/v1", "https://api.fireworks.ai/inference"),
        ("https://fireworks.test/inference/v1/", "https://fireworks.test/inference"),
        ("https://proxy.test/fireworks", "https://proxy.test/fireworks"),
    ],
)
def test_sdk_base_url_drops_the_openai_v1_suffix(api_base: str, expected: str) -> None:
    """The SDK appends /v1 itself, so only a trailing /v1 is removed from api_base."""
    assert _sdk_base_url(api_base) == expected


def test_init_client_points_the_sdk_at_the_default_inference_root() -> None:
    provider = FireworksProvider(api_key="test-key")
    assert str(provider.fireworks_client.base_url) == "https://api.fireworks.ai/inference/"
    assert str(provider.client.base_url) == "https://api.fireworks.ai/inference/v1/"


def test_init_client_defaults_the_sdk_timeout_to_the_openai_clients() -> None:
    """The SDK's own 60s read timeout is not used unless the caller asks for it."""
    provider = FireworksProvider(api_key="test-key")
    assert provider.fireworks_client.timeout == DEFAULT_TIMEOUT


def test_init_client_keeps_a_caller_timeout() -> None:
    provider = FireworksProvider(api_key="test-key", timeout=12.0)
    assert provider.fireworks_client.timeout == 12.0
    assert provider.client.timeout == 12.0


def test_convert_completion_params_sends_structured_output_as_json_schema() -> None:
    """The SDK has no parse() helper, so a Pydantic type becomes a JSON schema even without streaming."""
    params = CompletionParams(model_id=COMPLETION_MODEL, messages=USER_MESSAGES, response_format=_City)
    converted = FireworksProvider._convert_completion_params(params)
    assert converted["response_format"]["type"] == "json_schema"
    assert converted["response_format"]["json_schema"]["name"] == "_City"
    assert converted["response_format"]["json_schema"]["schema"]["properties"] == {
        "name": {"title": "Name", "type": "string"}
    }


def test_convert_completion_params_passes_a_dict_response_format_through() -> None:
    params = CompletionParams(
        model_id=COMPLETION_MODEL, messages=USER_MESSAGES, response_format={"type": "json_object"}
    )
    converted = FireworksProvider._convert_completion_params(params)
    assert converted["response_format"] == {"type": "json_object"}


def test_convert_completion_params_moves_stream_options_into_extra_body() -> None:
    """The SDK has no stream_options argument, and a caller's extra_body is kept."""
    params = CompletionParams(
        model_id=COMPLETION_MODEL, messages=USER_MESSAGES, stream=True, stream_options={"include_usage": True}
    )
    converted = FireworksProvider._convert_completion_params(params, extra_body={"top_k": 5})
    assert "stream_options" not in converted
    assert converted["extra_body"] == {"top_k": 5, "stream_options": {"include_usage": True}}


def test_convert_completion_params_without_stream_options_adds_no_extra_body() -> None:
    params = CompletionParams(model_id=COMPLETION_MODEL, messages=USER_MESSAGES, max_tokens=5)
    converted = FireworksProvider._convert_completion_params(params)
    assert "extra_body" not in converted
    assert converted["max_completion_tokens"] == 5


@pytest.mark.asyncio
async def test_acompletion_calls_the_sdk_and_normalizes_the_response() -> None:
    """A completion goes to <base>/v1/chat/completions and reasoning_content becomes reasoning."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completion_body(reasoning_content="thinking"))

    provider = _fireworks_with_transport(httpx.MockTransport(respond))
    result = await provider.acompletion(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": "Capital of France?", "extra_content": {"google": {"x": 1}}}],
        reasoning_effort="auto",
    )

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://fireworks.test/inference/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == COMPLETION_MODEL
    assert body["messages"] == [{"role": "user", "content": "Capital of France?"}]
    assert "reasoning_effort" not in body

    assert isinstance(result, ChatCompletion)
    assert result.object == "chat.completion"
    assert result.choices[0].message.content == "Paris"
    assert result.choices[0].message.reasoning == Reasoning(content="thinking")
    assert result.usage is not None
    assert result.usage.total_tokens == 13
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 4
    assert not hasattr(result, "perf_metrics")


@pytest.mark.asyncio
async def test_acompletion_parses_structured_output() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = _completion_body()
        body["choices"][0]["message"]["content"] = '{"name": "Paris"}'
        return httpx.Response(200, json=body)

    provider = _fireworks_with_transport(httpx.MockTransport(respond))
    result = await provider.acompletion(
        model=COMPLETION_MODEL, messages=[{"role": "user", "content": "Capital of France?"}], response_format=_City
    )

    assert json.loads(requests[0].content)["response_format"]["type"] == "json_schema"
    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed == _City(name="Paris")


@pytest.mark.asyncio
async def test_acompletion_streams_converted_chunks() -> None:
    """Streaming sends stream_options in the body and yields normalized chunks, usage included."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        content = _sse(
            _chunk({"role": "assistant", "reasoning_content": "hmm"}),
            _chunk({"content": "Paris"}, finish_reason="stop"),
            {
                **_chunk({}),
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            },
        )
        return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

    provider = _fireworks_with_transport(httpx.MockTransport(respond))
    stream = await provider.acompletion(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": "Capital of France?"}],
        stream=True,
        stream_options={"include_usage": True},
        reasoning_effort="low",
    )
    assert isinstance(stream, AsyncIterator)
    chunks = [chunk async for chunk in stream]

    body = json.loads(requests[0].content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["reasoning_effort"] == "low"
    assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0].choices[0].delta.reasoning == Reasoning(content="hmm")
    assert chunks[1].choices[0].delta.content == "Paris"
    assert chunks[1].choices[0].finish_reason == "stop"
    assert chunks[2].usage is not None
    assert chunks[2].usage.total_tokens == 13


@pytest.mark.asyncio
async def test_acompletion_maps_provider_errors() -> None:
    """A Fireworks SDK 404 surfaces as ModelNotFoundError with unified exceptions."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "Model not found"}})

    provider = _fireworks_with_transport(httpx.MockTransport(respond), unified_exceptions=True)
    with pytest.raises(ModelNotFoundError):
        await provider.acompletion(model="accounts/fireworks/models/nope", messages=[{"role": "user", "content": "hi"}])
