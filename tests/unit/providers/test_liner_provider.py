import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from any_llm.exceptions import AuthenticationError, ProviderError, RateLimitError, UnsupportedParameterError
from any_llm.providers.liner import LinerProvider
from any_llm.providers.liner.utils import _convert_references, _parse_event
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk

MESSAGES: list[Any] = [{"role": "user", "content": "What is Firefox?"}]

ANSWER = {
    "answer": "Firefox is a free, open-source web browser by Mozilla.",
    "references": [
        {
            "title": "Get Firefox for desktop and mobile",
            "url": "https://www.firefox.com/",
            "hostname": "firefox.com",
            "faviconUrl": "https://www.google.com/s2/favicons?domain=www.firefox.com&sz=128",
            "description": "Firefox is a free web browser backed by Mozilla.",
            "date": "",
        },
        {"title": "Firefox", "url": "https://en.wikipedia.org/wiki/Firefox", "imageUrl": "https://img.test/ff.png"},
    ],
}

STREAM = """\
:ping

event:data
data:{"type":"start","message_id":"msg-1"}

data:{"type":"start-step"}
data:{"type":"text-start","id":"text-1"}
data:{"type":"text-delta","id":"text-1","delta":"Firefox is"}
data:{"type":"text-delta","id":"text-1","delta":" a browser."}
data:{"type":"data-search-references","id":"r-1","data":{"references":[{"title":"Firefox","url":"https://www.firefox.com/"}]}}
data:{"type":"text-end","id":"text-1"}
data:{"type":"finish-step"}
data:{"type":"finish"}
data:[DONE]
"""


@pytest.fixture(autouse=True)
def clear_liner_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LinerProvider.ENV_API_BASE_NAME, raising=False)


def _provider(respond: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> LinerProvider:
    return LinerProvider(api_key="test-key", transport=httpx.MockTransport(respond), **kwargs)


def _recording(response: httpx.Response, requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response

    return respond


@pytest.mark.asyncio
async def test_completion_sends_the_agent_request_and_maps_references_to_citations() -> None:
    requests: list[httpx.Request] = []
    response = httpx.Response(200, json=ANSWER, headers={"x-request-id": "req-1"})
    provider = _provider(_recording(response, requests))

    result = await provider.acompletion(
        model="quick-answer", messages=MESSAGES, temperature=0.2, max_tokens=50, user="u-1"
    )

    assert isinstance(result, ChatCompletion)
    request = requests[0]
    assert str(request.url) == "https://platform.liner.com/api/v1/agents/quick-answer"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["accept"] == "application/json"
    assert json.loads(request.content) == {"messages": MESSAGES, "stream": False}
    assert result.id == "req-1"
    assert result.model == "quick-answer"
    choice = result.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert choice.message.content == ANSWER["answer"]
    assert choice.message.model_extra == {
        "citations": [
            {
                "url": "https://www.firefox.com/",
                "title": "Get Firefox for desktop and mobile",
                "description": "Firefox is a free web browser backed by Mozilla.",
                "favicon": "https://www.google.com/s2/favicons?domain=www.firefox.com&sz=128",
            },
            {"url": "https://en.wikipedia.org/wiki/Firefox", "title": "Firefox", "image": "https://img.test/ff.png"},
        ]
    }


@pytest.mark.asyncio
async def test_completion_sends_extra_keyword_arguments_and_honors_api_base() -> None:
    requests: list[httpx.Request] = []
    provider = _provider(_recording(httpx.Response(200, json=ANSWER), requests), api_base="https://liner.test/agents/")

    result = await provider.acompletion(model="search", messages=MESSAGES, mode="general")

    assert isinstance(result, ChatCompletion)
    assert str(requests[0].url) == "https://liner.test/agents/search"
    assert json.loads(requests[0].content) == {"messages": MESSAGES, "stream": False, "mode": "general"}
    assert result.id.startswith("chatcmpl-")


@pytest.mark.asyncio
@pytest.mark.parametrize("request_timeout", [None, 3.0])
async def test_completion_forwards_a_per_request_timeout(request_timeout: float | None) -> None:
    requests: list[httpx.Request] = []
    provider = _provider(_recording(httpx.Response(200, json=ANSWER), requests))

    await provider.acompletion(model="quick-answer", messages=MESSAGES, timeout=request_timeout)

    expected = request_timeout if request_timeout is not None else 60.0
    assert requests[0].extensions["timeout"]["read"] == expected
    assert "timeout" not in json.loads(requests[0].content)


@pytest.mark.asyncio
async def test_completion_flattens_text_parts() -> None:
    requests: list[httpx.Request] = []
    provider = _provider(_recording(httpx.Response(200, json=ANSWER), requests))
    messages: list[Any] = [
        {"role": "user", "content": [{"type": "text", "text": "What is"}, {"type": "text", "text": "Firefox?"}]},
        {"role": "assistant", "content": "A browser."},
        {"role": "user", "content": "Who makes it?"},
    ]

    await provider.acompletion(model="quick-answer", messages=messages)

    assert json.loads(requests[0].content)["messages"] == [
        {"role": "user", "content": "What is\nFirefox?"},
        {"role": "assistant", "content": "A browser."},
        {"role": "user", "content": "Who makes it?"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([{"role": "system", "content": "Be brief."}, *MESSAGES], UnsupportedParameterError),
        ([{"role": "tool", "content": "42", "tool_call_id": "c-1"}], UnsupportedParameterError),
        ([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}], UnsupportedParameterError),
        ([{"role": "assistant", "content": None}], UnsupportedParameterError),
        ([{"role": "user", "content": [{"type": "text"}]}], UnsupportedParameterError),
        ([{"role": "user", "content": [{"type": "text", "text": 42}]}], UnsupportedParameterError),
        ([*MESSAGES, {"role": "assistant", "content": "A browser."}], UnsupportedParameterError),
    ],
)
async def test_completion_refuses_messages_liner_cannot_take(messages: list[Any], error: type[Exception]) -> None:
    requests: list[httpx.Request] = []
    provider = _provider(_recording(httpx.Response(200, json=ANSWER), requests))

    with pytest.raises(error):
        await provider.acompletion(model="quick-answer", messages=messages)

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}]},
        {"tool_choice": "auto"},
        {"response_format": {"type": "json_object"}},
        {"parallel_tool_calls": False},
    ],
)
async def test_completion_refuses_parameters_that_change_the_result(params: dict[str, Any]) -> None:
    provider = _provider(lambda _request: httpx.Response(200, json=ANSWER))

    with pytest.raises(UnsupportedParameterError):
        await provider.acompletion(model="quick-answer", messages=MESSAGES, **params)


@pytest.mark.asyncio
async def test_completion_raises_on_a_response_without_an_answer() -> None:
    provider = _provider(lambda _request: httpx.Response(200, json={"references": []}))

    with pytest.raises(ProviderError):
        await provider.acompletion(model="quick-answer", messages=MESSAGES)


@pytest.mark.asyncio
async def test_completion_reports_an_http_error_by_status() -> None:
    body = {"error": {"code": "UNAUTHORIZED", "message": "Invalid API key", "retryable": False}}
    provider = _provider(lambda _request: httpx.Response(401, json=body), unified_exceptions=True)

    with pytest.raises(AuthenticationError) as exc_info:
        await provider.acompletion(model="quick-answer", messages=MESSAGES)

    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_stream_converts_events_to_chunks() -> None:
    requests: list[httpx.Request] = []
    response = httpx.Response(200, text=STREAM, headers={"x-request-id": "req-2"})
    provider = _provider(_recording(response, requests))

    stream = await provider.acompletion(model="quick-answer", messages=MESSAGES, stream=True)
    chunks: list[ChatCompletionChunk] = [chunk async for chunk in stream]

    assert requests[0].headers["accept"] == "text/event-stream"
    assert json.loads(requests[0].content) == {"messages": MESSAGES, "stream": True}
    assert [chunk.choices[0].delta.content for chunk in chunks] == ["Firefox is", " a browser.", None, None]
    assert [chunk.choices[0].delta.role for chunk in chunks] == ["assistant", None, None, None]
    assert chunks[2].choices[0].delta.model_extra == {
        "citations": [{"url": "https://www.firefox.com/", "title": "Firefox"}]
    }
    assert [chunk.choices[0].finish_reason for chunk in chunks] == [None, None, None, "stop"]
    assert {chunk.id for chunk in chunks} == {"req-2"}
    assert {chunk.model for chunk in chunks} == {"quick-answer"}


@pytest.mark.asyncio
async def test_stream_raises_on_an_error_event() -> None:
    error = {"code": "DOWNSTREAM_ERROR", "message": "Upstream service temporarily unavailable", "retryable": True}
    body = (
        'data:{"type":"text-delta","delta":"Fire"}\n'
        f'data:{{"type":"data-error","id":"data-error-1","data":{json.dumps(error)}}}\n'
        "data:[DONE]\n"
    )
    provider = _provider(lambda _request: httpx.Response(200, text=body))

    stream = await provider.acompletion(model="quick-answer", messages=MESSAGES, stream=True)
    with pytest.raises(ProviderError, match="Upstream service temporarily unavailable") as exc_info:
        [chunk async for chunk in stream]

    assert exc_info.value.code == "DOWNSTREAM_ERROR"


@pytest.mark.asyncio
async def test_stream_raises_on_an_error_event_without_details() -> None:
    body = 'data:{"type":"data-error","id":"data-error-1"}\ndata:[DONE]\n'
    provider = _provider(lambda _request: httpx.Response(200, text=body))

    stream = await provider.acompletion(model="quick-answer", messages=MESSAGES, stream=True)
    with pytest.raises(ProviderError, match="unknown error") as exc_info:
        [chunk async for chunk in stream]

    assert exc_info.value.code is None


@pytest.mark.asyncio
async def test_stream_reports_an_http_error_by_status() -> None:
    provider = _provider(
        lambda _request: httpx.Response(429, json={"error": {"message": "slow down"}}), unified_exceptions=True
    )

    stream = await provider.acompletion(model="quick-answer", messages=MESSAGES, stream=True)
    with pytest.raises(RateLimitError) as exc_info:
        [chunk async for chunk in stream]

    assert exc_info.value.status_code == 429


def test_sync_completion_runs_on_a_fresh_client_per_call() -> None:
    provider = _provider(lambda _request: httpx.Response(200, json=ANSWER))

    first = provider.completion(model="quick-answer", messages=MESSAGES)
    second = provider.completion(model="quick-answer", messages=MESSAGES)

    assert isinstance(first, ChatCompletion)
    assert isinstance(second, ChatCompletion)
    assert second.choices[0].message.content == ANSWER["answer"]


def test_convert_references_skips_items_without_a_url() -> None:
    references = [{"title": "no url"}, "not a dict", {"url": ""}, {"url": "https://a.test", "title": ""}]

    assert _convert_references(references) == [{"url": "https://a.test"}]
    assert _convert_references(None) == []


@pytest.mark.parametrize(
    ("line", "event"),
    [
        ('data:{"type":"finish"}', {"type": "finish"}),
        ('data: {"type":"finish"}', {"type": "finish"}),
        ("data:[DONE]", None),
        ("data:", None),
        ("data:[1]", None),
        ("event:data", None),
        (":ping", None),
    ],
)
def test_parse_event(line: str, event: dict[str, Any] | None) -> None:
    assert _parse_event(line) == event


@pytest.mark.parametrize("method", ["_convert_embedding_params", "_convert_list_models_response"])
def test_unsupported_conversions_raise(method: str) -> None:
    with pytest.raises(NotImplementedError):
        getattr(LinerProvider, method)(None)


def test_unsupported_embedding_response_raises() -> None:
    with pytest.raises(NotImplementedError):
        LinerProvider._convert_embedding_response(None)
