import dataclasses
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai.types.responses import ResponseCompactionItem, ResponseOutputMessage, ResponseOutputText
from openresponses_types import CompactionBody, ResponseResource
from pydantic import BaseModel

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model
from any_llm.types.responses import ParsedResponse, Response
from any_llm.utils.structured_output import parse_responses_output


def test_prompt_cache_key_capability_is_opt_in() -> None:
    assert BaseOpenAIProvider.PROMPT_CACHE_KEY_SUPPORT == "unsupported"
    assert OpenaiProvider.PROMPT_CACHE_KEY_SUPPORT == "supported"


def test_timeout_capability_is_native_for_openai_compatible_providers() -> None:
    """The OpenAI SDK takes `timeout` on the request, so every OpenAI-compatible provider inherits it."""
    assert BaseOpenAIProvider.TIMEOUT_SUPPORT == "native"
    assert OpenaiProvider.TIMEOUT_SUPPORT == "native"


@pytest.mark.asyncio
async def test_openai_messages_bridge_sends_typed_prompt_cache_key() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-5.6",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok", "refusal": None},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenaiProvider(api_key="test-key", http_client=http_client)
    try:
        await provider.amessages(
            model="gpt-5.6",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=100,
            prompt_cache_key="tenant-1",
        )
    finally:
        await http_client.aclose()

    assert len(requests) == 1
    assert json.loads(requests[0].content)["prompt_cache_key"] == "tenant-1"


class _City(BaseModel):
    city_name: str


@dataclasses.dataclass
class _CityDataclass:
    city_name: str


class _ResponsesProvider(BaseOpenAIProvider):
    SUPPORTS_RESPONSES = True
    PROVIDER_NAME = "ResponsesProvider"
    ENV_API_KEY_NAME = "TEST_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://example.com"
    API_BASE = "https://api.example.com/v1"


def _make_openai_response(text: str) -> Response:
    message = ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return Response(
        id="resp-1",
        created_at=0,
        model="test-model",
        object="response",
        output=[message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _make_openai_compaction_response() -> Response:
    compaction = ResponseCompactionItem(
        id="cmp-1",
        type="compaction",
        encrypted_content="opaque-compacted-context",
    )
    return Response(
        id="resp-compaction",
        created_at=0,
        model="gpt-5.3-codex",
        object="response",
        output=[compaction],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def _make_populated_openai_compaction_response() -> Response:
    """A compaction response carrying every field ResponseResource requires.

    Built from a wire-shape payload rather than SDK kwargs: ResponseResource
    requires `store`, `presence_penalty` and `frequency_penalty`, which openai's
    Response does not declare as fields and only carries as extras.

    The sparse fixtures above leave optional scalars unset, so they fail
    ResponseResource validation on those fields regardless of compaction support
    and exercise the fallback path rather than the normalization path.
    """
    return Response.model_validate(
        {
            "id": "resp-compaction",
            "created_at": 0,
            "model": "gpt-5.3-codex",
            "object": "response",
            "output": [
                {
                    "id": "cmp-1",
                    "type": "compaction",
                    "encrypted_content": "opaque-compacted-context",
                },
                {
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Resumed.", "annotations": [], "logprobs": []},
                    ],
                },
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "status": "completed",
            "store": False,
            "background": False,
            "truncation": "disabled",
            "metadata": {},
            "temperature": 1.0,
            "top_p": 1.0,
            "top_logprobs": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "service_tier": "auto",
            "text": {"format": {"type": "text"}},
        }
    )


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_basemodel_uses_parse(mock_openai_class: MagicMock) -> None:
    parsed = parse_responses_output(_make_openai_response('{"city_name": "Paris"}'), _City)

    mock_client = AsyncMock()
    mock_client.responses.parse = AsyncMock(return_value=parsed)
    mock_client.responses.create = AsyncMock()
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    result = await provider.aresponses(model="gpt-4o", input_data="capital of France?", response_format=_City)

    mock_client.responses.parse.assert_awaited_once()
    assert mock_client.responses.parse.call_args.kwargs["text_format"] is _City
    mock_client.responses.create.assert_not_called()
    assert isinstance(result, ParsedResponse)
    assert isinstance(result.output_parsed, _City)
    assert result.output_parsed.city_name == "Paris"


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_dataclass_uses_create_and_is_parsed(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=_make_openai_response('{"city_name": "Paris"}'))
    mock_client.responses.parse = AsyncMock()
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    result = await provider.aresponses(model="gpt-4o", input_data="capital of France?", response_format=_CityDataclass)

    mock_client.responses.parse.assert_not_called()
    create_kwargs = mock_client.responses.create.call_args.kwargs
    assert create_kwargs["text"]["format"]["type"] == "json_schema"
    assert "response_format" not in create_kwargs
    assert isinstance(result, ParsedResponse)
    assert isinstance(result.output_parsed, _CityDataclass)
    assert result.output_parsed.city_name == "Paris"


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_dict_response_format_is_passed_through_unparsed(mock_openai_class: MagicMock) -> None:
    raw = _make_openai_response('{"city_name": "Paris"}')
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=raw)
    mock_client.responses.parse = AsyncMock()
    mock_openai_class.return_value = mock_client

    response_format = {"type": "json_schema", "name": "City", "schema": {"type": "object"}}

    provider = _ResponsesProvider(api_key="key")
    result = await provider.aresponses(model="gpt-4o", input_data="hi", response_format=response_format)

    mock_client.responses.parse.assert_not_called()
    assert mock_client.responses.create.call_args.kwargs["text"] == {"format": response_format}
    assert not isinstance(result, ParsedResponse)


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_stream_with_response_format_raises(mock_openai_class: MagicMock) -> None:
    mock_openai_class.return_value = AsyncMock()
    provider = _ResponsesProvider(api_key="key")

    with pytest.raises(ValueError, match="stream is not supported for response_format"):
        await provider.aresponses(model="gpt-4o", input_data="hi", response_format=_City, stream=True)


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_without_response_format_returns_response(mock_openai_class: MagicMock) -> None:
    """No response_format: neither parse nor text.format is used, and the raw Response is returned."""
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=_make_openai_response('{"city_name": "Paris"}'))
    mock_client.responses.parse = AsyncMock()
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    result = await provider.aresponses(model="gpt-4o", input_data="hi")

    mock_client.responses.parse.assert_not_called()
    assert "text" not in mock_client.responses.create.call_args.kwargs
    assert "context_management" not in mock_client.responses.create.call_args.kwargs
    assert isinstance(result, Response)
    assert not isinstance(result, ParsedResponse)


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_context_management_preserves_compaction_item(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=_make_openai_compaction_response())
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    context_management = [{"type": "compaction", "compact_threshold": 200_000}]
    result = await provider.aresponses(
        model="gpt-5.3-codex",
        input_data="Continue the coding task.",
        context_management=context_management,
    )

    assert mock_client.responses.create.call_args.kwargs["context_management"] == context_management
    # This fixture omits optional scalars (status, store, text, ...), so it fails
    # ResponseResource validation on those and falls back to the raw Response.
    # See test_aresponses_normalizes_compaction_item for the normalization path.
    assert isinstance(result, Response)
    assert isinstance(result.output[0], ResponseCompactionItem)
    assert result.output[0].encrypted_content == "opaque-compacted-context"


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_normalizes_compaction_item(mock_openai_class: MagicMock) -> None:
    """A fully populated compaction response normalizes to a ResponseResource.

    Requires openresponses-types >= 2.4.0, where ResponseResource.output accepts
    CompactionBody. Against earlier releases the compaction item failed validation
    and aresponses fell back to returning the raw OpenAI Response.
    """
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=_make_populated_openai_compaction_response())
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    result = await provider.aresponses(model="gpt-5.3-codex", input_data="Continue the coding task.")

    assert isinstance(result, ResponseResource)
    assert [item.type for item in result.output] == ["compaction", "message"]
    compaction = result.output[0]
    assert isinstance(compaction, CompactionBody)
    assert compaction.encrypted_content == "opaque-compacted-context"


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_replays_compaction_item_unchanged(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=_make_openai_response("Done"))
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    compaction_input = _make_openai_compaction_response().output[0].model_dump(exclude_none=True)
    input_data = [compaction_input, {"role": "user", "content": "Continue."}]
    await provider.aresponses(model="gpt-5.3-codex", input_data=input_data)

    assert mock_client.responses.create.call_args.kwargs["input"] == input_data


@pytest.mark.asyncio
@patch("any_llm.providers.openai.base.AsyncOpenAI")
async def test_aresponses_forwards_extra_body(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_client.responses.create = AsyncMock(return_value=_make_openai_response('{"city_name": "Paris"}'))
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    extra_body = {
        "client_metadata": {"x-codex-turn-metadata": {"session_id": "session_123"}},
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
                "internal_chat_message_metadata_passthrough": {"turn_id": "turn_123"},
            }
        ],
    }

    await provider.aresponses(model="gpt-4o", input_data="hi", extra_body=extra_body)

    assert mock_client.responses.create.call_args.kwargs["extra_body"] == extra_body


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_responses_sync_basemodel_returns_parsed(mock_openai_class: MagicMock) -> None:
    """The synchronous wrapper must return the ParsedResponse (a Response subclass), not iterate it."""
    parsed = parse_responses_output(_make_openai_response('{"city_name": "Paris"}'), _City)

    mock_client = AsyncMock()
    mock_client.responses.parse = AsyncMock(return_value=parsed)
    mock_openai_class.return_value = mock_client

    provider = _ResponsesProvider(api_key="key")
    result = provider.responses(model="gpt-4o", input_data="capital of France?", response_format=_City)

    assert isinstance(result, ParsedResponse)
    assert isinstance(result.output_parsed, _City)
    assert result.output_parsed.city_name == "Paris"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_list_models_returns_model_list_when_supported(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://api.example.com/v1"

    mock_model_data = [
        Model(id="gpt-3.5-turbo", object="model", created=1677610602, owned_by="openai"),
        Model(id="gpt-4", object="model", created=1687882411, owned_by="openai"),
    ]

    mock_client = AsyncMock()
    mock_client.models.list.return_value.data = mock_model_data
    mock_openai_class.return_value = mock_client

    provider = ListModelsProvider(api_key="test-key", api_base="https://custom.api.com/v1")

    result = provider.list_models()

    assert result == mock_model_data
    mock_openai_class.assert_called_once_with(base_url="https://custom.api.com/v1", api_key="test-key")
    mock_client.models.list.assert_called_once_with()


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_list_models_uses_default_api_base_when_not_configured(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"
        API_BASE = "https://api.default.com/v1"

    mock_client = AsyncMock()
    mock_client.models.list.return_value.data = []
    mock_openai_class.return_value = mock_client

    provider = ListModelsProvider(api_key="test-key")

    provider.list_models()

    mock_openai_class.assert_called_once_with(base_url="https://api.default.com/v1", api_key="test-key")


@patch(
    "any_llm.providers.openai.base.AsyncOpenAI",
)
def test_list_models_passes_kwargs_to_client(mock_openai_class: MagicMock) -> None:
    class ListModelsProvider(BaseOpenAIProvider):
        SUPPORTS_LIST_MODELS = True
        PROVIDER_NAME = "ListModelsProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    mock_client = AsyncMock()
    mock_client.models.list.return_value.data = []
    mock_openai_class.return_value = mock_client

    provider = ListModelsProvider(api_key="test-key")

    provider.list_models(limit=10, after="model-123")

    mock_client.models.list.assert_called_once_with(limit=10, after="model-123")


@pytest.mark.asyncio
async def test_stream_with_dict_response_format_uses_create() -> None:
    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "TestProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    response_format = {"type": "json_object"}

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai_class:
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        provider = TestProvider(api_key="test-key")

        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format=response_format,
            )
        )

        mock_client.chat.completions.create.assert_awaited_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            response_format=response_format,
        )
        mock_client.chat.completions.parse.assert_not_called()


@pytest.mark.asyncio
async def test_stream_with_basemodel_response_format_uses_create_json_schema() -> None:
    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "TestProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai_class:
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        provider = TestProvider(api_key="test-key")

        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format=_City,
            )
        )

        create_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert create_kwargs["stream"] is True
        assert create_kwargs["response_format"]["type"] == "json_schema"
        assert create_kwargs["response_format"]["json_schema"]["name"] == "_City"
        assert create_kwargs["response_format"]["json_schema"]["schema"]["properties"] == {
            "city_name": {"title": "City Name", "type": "string"}
        }
        mock_client.chat.completions.parse.assert_not_called()


def test_base_provider_maps_max_tokens_to_max_completion_tokens() -> None:
    params = CompletionParams(model_id="model", messages=[{"role": "user", "content": "hi"}], max_tokens=8192)
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192


def test_base_provider_preserves_explicit_max_completion_tokens() -> None:
    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=4096,
    )
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 4096


def test_base_provider_max_completion_tokens_takes_precedence_over_max_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8192,
        max_completion_tokens=4096,
    )

    any_llm_logger = logging.getLogger("any_llm")
    any_llm_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            result = BaseOpenAIProvider._convert_completion_params(params)
    finally:
        any_llm_logger.propagate = False

    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 4096
    assert "Ignoring max_tokens (8192) in favor of max_completion_tokens (4096)" in caplog.text


def test_base_provider_no_max_tokens_passes_through_unchanged() -> None:
    params = CompletionParams(model_id="model", messages=[{"role": "user", "content": "hi"}], temperature=0.5)
    result = BaseOpenAIProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert "max_completion_tokens" not in result
    assert result["temperature"] == 0.5


def test_base_provider_max_tokens_via_kwargs_also_remapped() -> None:
    params = CompletionParams(model_id="model", messages=[{"role": "user", "content": "hi"}])
    result = BaseOpenAIProvider._convert_completion_params(params, max_tokens=1024)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 1024


def test_base_provider_converts_dataclass_response_format_to_json_schema() -> None:
    """Test that plain dataclasses are converted to JSON schema dicts."""
    from dataclasses import dataclass

    @dataclass
    class TestOutput:
        name: str
        value: int

    params = CompletionParams(
        model_id="model",
        messages=[{"role": "user", "content": "hi"}],
        response_format=TestOutput,
    )
    result = BaseOpenAIProvider._convert_completion_params(params)

    assert result["response_format"]["type"] == "json_schema"
    assert result["response_format"]["json_schema"]["name"] == "TestOutput"
    assert "properties" in result["response_format"]["json_schema"]["schema"]


@pytest.mark.asyncio
async def test_acompletion_with_dataclass_uses_create_not_parse() -> None:
    """Test that plain dataclasses use .create() instead of .parse()."""
    from dataclasses import dataclass

    @dataclass
    class TestOutput:
        name: str

    class TestProvider(BaseOpenAIProvider):
        PROVIDER_NAME = "TestProvider"
        ENV_API_KEY_NAME = "TEST_API_KEY"
        PROVIDER_DOCUMENTATION_URL = "https://example.com"

    with patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai_class:
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=MagicMock())

        provider = TestProvider(api_key="test-key")
        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            )
        )

        mock_client.chat.completions.create.assert_called_once()
        mock_client.chat.completions.parse.assert_not_called()
