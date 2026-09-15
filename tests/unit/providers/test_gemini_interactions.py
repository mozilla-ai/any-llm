import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import types
from google.genai._gaos.types.interactions import Interaction
from google.genai.interactions import (
    Error,
    ImageContent,
    ModelOutputStep,
    TextContent,
    ThoughtStep,
    Usage,
)
from openai.types.responses import (
    ResponseOutputMessage,
)

from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.base import GoogleProvider
from any_llm.providers.gemini.interactions import (
    convert_interaction_to_response,
    convert_responses_params,
)
from any_llm.providers.vertexai import VertexaiProvider
from any_llm.types.responses import Response, ResponsesParams


def _interaction(
    *,
    status: str = "completed",
    created: str | None = "2026-01-02T03:04:05Z",
    steps: list[object] | None = None,
    usage: Usage | None = None,
) -> Interaction:
    if steps is None:
        steps = [ModelOutputStep(content=[TextContent(text="Hello")])]
    if usage is None:
        usage = Usage(
            total_input_tokens=4,
            total_output_tokens=2,
            total_tokens=6,
            total_cached_tokens=1,
            total_thought_tokens=3,
        )
    return Interaction.model_validate(
        {
            "id": "int-123",
            "status": status,
            "model": "gemini-3.8-flash",
            "created": created,
            "previous_interaction_id": "int-previous",
            "system_instruction": "Be concise",
            "labels": {"team": "sdk"},
            "steps": steps,
            "usage": usage,
        }
    )


def test_gemini_enables_responses_without_changing_shared_google_provider() -> None:
    assert GeminiProvider.SUPPORTS_RESPONSES is True
    assert GoogleProvider.SUPPORTS_RESPONSES is False
    assert VertexaiProvider.SUPPORTS_RESPONSES is False


def test_convert_interaction_maps_text_status_metadata_and_usage() -> None:
    response = convert_interaction_to_response(_interaction())

    assert isinstance(response, Response)
    assert response.id == "int-123"
    assert response.status == "completed"
    assert response.model == "gemini-3.8-flash"
    assert response.created_at == 1767323045.0
    assert response.previous_response_id == "int-previous"
    assert response.instructions == "Be concise"
    assert response.metadata == {"team": "sdk"}
    assert response.output_text == "Hello"
    message = response.output[0]
    assert isinstance(message, ResponseOutputMessage)
    assert message.id == "msg-int-123-0"
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 6
    assert response.usage.input_tokens_details.cached_tokens == 1
    assert response.usage.output_tokens_details.reasoning_tokens == 3


def test_convert_interaction_preserves_explicit_zero_total_usage() -> None:
    usage = Usage(total_input_tokens=2, total_output_tokens=3, total_tokens=0)
    response = convert_interaction_to_response(_interaction(usage=usage))

    assert response.usage is not None
    assert response.usage.total_tokens == 0


def test_convert_interaction_preserves_absent_usage() -> None:
    interaction = _interaction()
    interaction.usage = None

    assert convert_interaction_to_response(interaction).usage is None


def test_convert_interaction_preserves_empty_text_output() -> None:
    response = convert_interaction_to_response(_interaction(steps=[ModelOutputStep(content=[TextContent(text="")])]))

    assert len(response.output) == 1
    assert response.output_text == ""


def test_convert_interaction_ignores_non_output_steps() -> None:
    interaction = _interaction(
        steps=[
            {"type": "future_step", "future": True},
            ModelOutputStep(content=[TextContent(text="kept")]),
        ]
    )

    response = convert_interaction_to_response(interaction)

    assert response.output_text == "kept"
    assert len(response.output) == 1
    assert response.output[0].id == "msg-int-123-0"


@pytest.mark.parametrize(
    "content",
    [
        [ImageContent(data="aW1hZ2U=", mime_type="image/png")],
        [TextContent(text="partial"), ImageContent(data="aW1hZ2U=", mime_type="image/png")],
        [TextContent(text="partial"), {"type": "future_content", "future": True}],
    ],
)
def test_convert_interaction_rejects_non_text_model_output(content: list[object]) -> None:
    step = ModelOutputStep.model_validate({"content": content})

    with pytest.raises(ProviderError, match="non-text model output"):
        convert_interaction_to_response(_interaction(steps=[step]))


def test_convert_interaction_keeps_text_after_thought_output() -> None:
    response = convert_interaction_to_response(
        _interaction(steps=[ThoughtStep(signature="opaque"), ModelOutputStep(content=[TextContent(text="25")])])
    )

    assert response.output_text == "25"
    assert len(response.output) == 1
    assert response.output[0].id == "msg-int-123-0"


@pytest.mark.parametrize("identifiers", [("first", "second"), (None, None)])
def test_convert_interaction_message_ids_are_unique(identifiers: tuple[str | None, str | None]) -> None:
    responses = [
        convert_interaction_to_response(
            _interaction(
                steps=[
                    ModelOutputStep(content=[TextContent(text="first part")]),
                    ModelOutputStep(content=[TextContent(text="second part")]),
                ]
            ).model_copy(update={"id": identifier})
        )
        for identifier in identifiers
    ]

    assert len({item.id for response in responses for item in response.output}) == 4
    for response in responses:
        assert [item.id for item in response.output] == [f"msg-{response.id}-0", f"msg-{response.id}-1"]


@pytest.mark.parametrize(
    ("error", "expected_message"),
    [
        (Error(code="gateway_timeout", message="deadline expired"), "deadline expired"),
        (Error(code="gateway_timeout"), "gateway_timeout"),
        (Error(), "Gemini interaction failed"),
    ],
)
def test_convert_interaction_maps_provider_error_without_raw_side_channel(error: Error, expected_message: str) -> None:
    interaction = _interaction(status="failed", steps=[])
    interaction.errors = [error]

    response = convert_interaction_to_response(interaction)

    assert response.error is not None
    assert response.error.code == "server_error"
    assert response.error.message == expected_message
    assert not any(name.startswith("gemini_") for name in response.model_dump())


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="Process timezone control requires time.tzset")
@pytest.mark.parametrize("timezone", ["UTC0", "EST5"])
@pytest.mark.parametrize(
    "created",
    ["2026-01-02T03:04:05", "2026-01-02T03:04:05Z", "2026-01-02T08:34:05+05:30"],
)
def test_convert_interaction_timestamp_is_independent_of_process_timezone(
    monkeypatch: pytest.MonkeyPatch, timezone: str, created: str
) -> None:
    try:
        with monkeypatch.context() as environment:
            environment.setenv("TZ", timezone)
            time.tzset()
            response = convert_interaction_to_response(_interaction(created=created))
    finally:
        time.tzset()

    assert response.created_at == 1767323045.0


@pytest.mark.parametrize("created", ["invalid", None, ""])
def test_convert_interaction_handles_unknown_status_and_invalid_timestamp(created: str | None) -> None:
    response = convert_interaction_to_response(_interaction(status="future_status", created=created))

    assert response.status == "in_progress"
    assert response.created_at == 0.0


@pytest.mark.parametrize(
    ("gemini_status", "expected_status"),
    [
        ("queued", "queued"),
        ("requires_action", "incomplete"),
        ("budget_exceeded", "incomplete"),
    ],
)
def test_convert_interaction_normalizes_extended_sdk_statuses(
    gemini_status: str,
    expected_status: str,
) -> None:
    assert convert_interaction_to_response(_interaction(status=gemini_status)).status == expected_status


def test_convert_responses_params_maps_only_reviewed_text_subset() -> None:
    params = ResponsesParams(
        model="gemini-3.8-flash",
        input="Hello",
        instructions="",
        max_output_tokens=0,
        stream=True,
    )

    assert convert_responses_params(params, "gemini", api_version="v1") == {
        "api_version": "v1",
        "model": "gemini-3.8-flash",
        "input": "Hello",
        "system_instruction": "",
        "generation_config": {"max_output_tokens": 0},
        "stream": True,
    }


def test_convert_responses_params_rejects_non_string_input() -> None:
    params = ResponsesParams(model="gemini-3.8-flash", input=[{"type": "input_text", "text": "Hello"}])

    with pytest.raises(UnsupportedParameterError, match="input"):
        convert_responses_params(params, "gemini", api_version="v1")


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("tools", [{"type": "function", "name": "lookup"}]),
        ("reasoning", {"effort": "low"}),
        ("response_format", {"type": "json_object"}),
        ("background", True),
        ("temperature", 0.2),
        ("store", False),
        ("metadata", {}),
        ("previous_response_id", "int-previous"),
    ],
)
def test_convert_responses_params_rejects_unimplemented_surface(parameter: str, value: object) -> None:
    params = ResponsesParams.model_validate({"model": "gemini-3.8-flash", "input": "Hello", parameter: value})

    with pytest.raises(UnsupportedParameterError, match=parameter):
        convert_responses_params(params, "gemini", api_version="v1")


@pytest.mark.asyncio
async def test_aresponses_defaults_only_interactions_requests_to_v1() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        result = await provider.aresponses(
            "gemini-3.8-flash",
            "Hello",
            instructions="Be concise",
            timeout=1.5,
        )

    assert isinstance(result, Response)
    assert result.output_text == "Hello"
    client_class.assert_called_once_with(api_key="test-key")
    client.aio.interactions.create.assert_awaited_once_with(
        api_version="v1",
        model="gemini-3.8-flash",
        input="Hello",
        system_instruction="Be concise",
        timeout=1.5,
    )


@pytest.mark.asyncio
async def test_aresponses_preserves_explicit_v1beta_client_configuration() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key", http_options={"api_version": "v1beta"})
        await provider.aresponses("gemini-3.8-flash", "Hello")

    assert client_class.call_args.kwargs["http_options"] == {"api_version": "v1beta"}
    assert client.aio.interactions.create.await_args.kwargs["api_version"] == "v1beta"


@pytest.mark.asyncio
async def test_aresponses_rejects_openai_extra_body() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client"):
        provider = GeminiProvider(api_key="test-key")
        with pytest.raises(UnsupportedParameterError, match="extra_body"):
            await provider.aresponses(
                "gemini-3.8-flash",
                "Hello",
                extra_body={"future": True},
            )


@pytest.mark.asyncio
async def test_aresponses_rejects_streaming_before_io() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(UnsupportedParameterError, match="stream"):
            await provider.aresponses("gemini-3.8-flash", "Hello", stream=True)

    client_class.return_value.aio.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_aresponses_forwards_sdk_transport_parameters() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        await provider._aresponses(
            ResponsesParams(model="gemini-3.8-flash", input="Hello"),
            extra_headers={"x-request-id": "request-123"},
            extra_query={"trace": "enabled"},
        )

    assert client.aio.interactions.create.await_args.kwargs["extra_headers"] == {"x-request-id": "request-123"}
    assert client.aio.interactions.create.await_args.kwargs["extra_query"] == {"trace": "enabled"}


@pytest.mark.asyncio
async def test_aresponses_rejects_unknown_transport_parameter_before_io() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key")

        with pytest.raises(UnsupportedParameterError, match="future_transport"):
            await provider._aresponses(
                ResponsesParams(model="gemini-3.8-flash", input="Hello"),
                future_transport=True,
            )

    client.aio.interactions.create.assert_not_called()


@pytest.mark.asyncio
async def test_aresponses_ignores_none_transport_parameters() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        client.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        await provider._aresponses(
            ResponsesParams(model="gemini-3.8-flash", input="Hello"),
            extra_headers=None,
            extra_query=None,
            future_transport=None,
        )

    assert "extra_headers" not in client.aio.interactions.create.await_args.kwargs
    assert "extra_query" not in client.aio.interactions.create.await_args.kwargs
    assert "future_transport" not in client.aio.interactions.create.await_args.kwargs


@pytest.mark.parametrize("total", [None, 0, 346])
@pytest.mark.asyncio
async def test_real_sdk_serializes_stable_interactions_path_and_body(total: int | None) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "int-123",
                "status": "completed",
                "model": "gemini-3.8-flash",
                "steps": [
                    {"type": "thought", "signature": "opaque"},
                    {"type": "model_output", "content": [{"type": "text", "text": "Hello"}]},
                ],
                "usage": {
                    "total_input_tokens": 11,
                    "total_output_tokens": 90,
                    "total_thought_tokens": 245,
                    "total_tokens": total,
                },
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        response = await provider.aresponses(
            "gemini-3.8-flash",
            "Hello",
            instructions="",
            max_output_tokens=0,
        )
    finally:
        await http_client.aclose()

    assert isinstance(response, Response)
    assert response.output_text == "Hello"
    assert response.output[0].id == "msg-int-123-0"
    assert response.usage is not None
    assert response.usage.output_tokens == 335
    assert response.usage.output_tokens_details.reasoning_tokens == 245
    assert response.usage.total_tokens == (346 if total is None else total)
    assert len(requests) == 1
    assert str(requests[0].url) == "https://example.test/v1/interactions"
    assert requests[0].headers["x-goog-api-key"] == "test-key"
    assert json.loads(requests[0].content) == {
        "input": "Hello",
        "model": "gemini-3.8-flash",
        "generation_config": {"max_output_tokens": 0},
        "system_instruction": "",
    }


@pytest.mark.asyncio
async def test_real_sdk_http_error_uses_unified_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            json={"error": {"code": "INVALID_ARGUMENT", "message": "invalid input"}},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        with pytest.raises(InvalidRequestError) as raised:
            await provider.aresponses("gemini-3.8-flash", "Hello")
    finally:
        await http_client.aclose()

    assert raised.value.status_code == 400
    assert raised.value.code == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_real_sdk_timeout_uses_unified_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    async def handler(request: httpx.Request) -> httpx.Response:
        message = "read timed out"
        raise httpx.ReadTimeout(message, request=request)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = GeminiProvider(
            api_key="test-key",
            api_base="https://example.test",
            http_options=types.HttpOptions(httpx_async_client=http_client),
        )
        with pytest.raises(ProviderError) as raised:
            await provider.aresponses("gemini-3.8-flash", "Hello", timeout=0.1)
    finally:
        await http_client.aclose()

    assert raised.value.status_code is None
    original = raised.value.original_exception
    assert original is not None
    assert type(original).__name__ == "APITimeoutError"
    assert isinstance(original.__cause__, httpx.ReadTimeout)


def test_responses_calls_interactions_synchronously() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client_class.return_value.aio.interactions.create = AsyncMock(return_value=_interaction())
        provider = GeminiProvider(api_key="test-key")
        response = provider.responses("gemini-3.8-flash", "Hello")

    assert isinstance(response, Response)
    assert response.output_text == "Hello"
