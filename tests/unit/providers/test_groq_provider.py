from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.completion import CompletionParams

groq = pytest.importorskip("groq")


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(api_key="test-api-key")

    with pytest.raises(UnsupportedParameterError, match="stream and response_format"):
        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_unsupported_max_tool_calls_parameter() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(api_key="test-api-key")

    with pytest.raises(UnsupportedParameterError):
        await provider.aresponses("test_model", "test_data", max_tool_calls=3)


@pytest.mark.asyncio
async def test_completion_with_response_format_basemodel() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    class TestOutput(BaseModel):
        foo: str

    with (
        patch("any_llm.providers.groq.groq.AsyncGroq") as mocked_groq,
        patch("any_llm.providers.groq.groq.to_chat_completion") as mocked_to_chat_completion,
    ):
        provider = GroqProvider(api_key="test-api-key")

        mock_response = Mock()
        mocked_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        mocked_to_chat_completion.return_value = mock_response

        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            ),
        )
        call_kwargs = mocked_groq.return_value.chat.completions.create.call_args[1]

        assert call_kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "TestOutput",
                "schema": {
                    "properties": {"foo": {"title": "Foo", "type": "string"}},
                    "required": ["foo"],
                    "title": "TestOutput",
                    "type": "object",
                },
            },
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Groq API calls."""
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    with (
        patch("any_llm.providers.groq.groq.AsyncGroq") as mocked_groq,
        patch("any_llm.providers.groq.groq.to_chat_completion") as mocked_to_chat_completion,
    ):
        provider = GroqProvider(api_key="test-api-key")

        mock_response = Mock()
        mocked_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        mocked_to_chat_completion.return_value = mock_response

        await provider._acompletion(
            CompletionParams(
                model_id="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            ),
        )

        call_kwargs = mocked_groq.return_value.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs


def test_stream_options_filtered_out() -> None:
    """stream_options is an OpenAI-only knob (set by the Messages bridge for
    streaming usage); the Groq SDK rejects it, so it must be dropped."""
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    result = GroqProvider._convert_completion_params(
        CompletionParams(
            model_id="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
    )
    assert "stream_options" not in result


def test_to_chat_completion_extracts_cached_tokens() -> None:
    """Test that cached tokens from Groq usage are extracted into prompt_tokens_details."""
    from groq.types.chat import ChatCompletion as GroqChatCompletion

    from any_llm.providers.groq.utils import to_chat_completion

    response = GroqChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4641,
                "completion_tokens": 1817,
                "total_tokens": 6458,
                "prompt_tokens_details": {"cached_tokens": 4608},
            },
        }
    )

    result = to_chat_completion(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 4641
    assert result.usage.completion_tokens == 1817
    assert result.usage.total_tokens == 6458
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 4608


def test_to_chat_completion_without_cached_tokens() -> None:
    """Test that prompt_tokens_details is None when no cached tokens are present."""
    from groq.types.chat import ChatCompletion as GroqChatCompletion

    from any_llm.providers.groq.utils import to_chat_completion

    response = GroqChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
    )

    result = to_chat_completion(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.prompt_tokens_details is None


@pytest.mark.asyncio
async def test_completion_with_response_format_dataclass() -> None:
    pytest.importorskip("groq")
    from dataclasses import dataclass

    from any_llm.providers.groq.groq import GroqProvider

    @dataclass
    class TestOutput:
        foo: str

    with (
        patch("any_llm.providers.groq.groq.AsyncGroq") as mocked_groq,
        patch("any_llm.providers.groq.groq.to_chat_completion") as mocked_to_chat_completion,
    ):
        provider = GroqProvider(api_key="test-api-key")

        mock_response = Mock()
        mocked_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        mocked_to_chat_completion.return_value = mock_response

        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            ),
        )
        call_kwargs = mocked_groq.return_value.chat.completions.create.call_args[1]

        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "TestOutput"
        assert "properties" in call_kwargs["response_format"]["json_schema"]["schema"]


def test_streaming_chunk_extracts_cached_tokens() -> None:
    """Test that streaming chunks correctly extract cached tokens from Groq usage."""
    from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk

    from any_llm.providers.groq.utils import _create_openai_chunk_from_groq_chunk

    chunk = GroqChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4641,
                "completion_tokens": 1817,
                "total_tokens": 6458,
                "prompt_tokens_details": {"cached_tokens": 4608},
            },
        }
    )

    result = _create_openai_chunk_from_groq_chunk(chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 4641
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 4608


def _make_openai_response(text: str):  # type: ignore[no-untyped-def]
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from any_llm.types.responses import Response

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


@pytest.mark.asyncio
@patch("any_llm.providers.groq.groq.AsyncOpenAI")
async def test_groq_aresponses_basemodel_uses_parse(mock_openai_class: Mock) -> None:
    from any_llm.providers.groq.groq import GroqProvider
    from any_llm.types.responses import ParsedResponse
    from any_llm.utils.structured_output import parse_responses_output

    class City(BaseModel):
        city_name: str

    parsed = parse_responses_output(_make_openai_response('{"city_name": "Paris"}'), City)

    client = AsyncMock()
    client.responses.parse = AsyncMock(return_value=parsed)
    client.responses.create = AsyncMock()
    mock_openai_class.return_value = client

    provider = GroqProvider(api_key="test-api-key")
    result = await provider.aresponses("openai/gpt-oss-20b", "capital of France?", response_format=City)

    client.responses.parse.assert_awaited_once()
    assert client.responses.parse.call_args.kwargs["text_format"] is City
    client.responses.create.assert_not_called()
    assert isinstance(result, ParsedResponse)
    assert result.output_parsed is not None
    assert result.output_parsed.city_name == "Paris"


@pytest.mark.asyncio
@patch("any_llm.providers.groq.groq.AsyncOpenAI")
async def test_groq_aresponses_dataclass_uses_create_and_is_parsed(mock_openai_class: Mock) -> None:
    import dataclasses

    from any_llm.providers.groq.groq import GroqProvider
    from any_llm.types.responses import ParsedResponse

    @dataclasses.dataclass
    class City:
        city_name: str

    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=_make_openai_response('{"city_name": "Paris"}'))
    client.responses.parse = AsyncMock()
    mock_openai_class.return_value = client

    provider = GroqProvider(api_key="test-api-key")
    result = await provider.aresponses("openai/gpt-oss-20b", "capital of France?", response_format=City)

    client.responses.parse.assert_not_called()
    assert client.responses.create.call_args.kwargs["text"]["format"]["type"] == "json_schema"
    assert isinstance(result, ParsedResponse)
    assert result.output_parsed is not None
    assert result.output_parsed.city_name == "Paris"


@pytest.mark.asyncio
@patch("any_llm.providers.groq.groq.AsyncOpenAI")
async def test_groq_aresponses_dict_response_format_sets_text_format(mock_openai_class: Mock) -> None:
    from any_llm.providers.groq.groq import GroqProvider
    from any_llm.types.responses import ParsedResponse

    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=_make_openai_response("{}"))
    client.responses.parse = AsyncMock()
    mock_openai_class.return_value = client

    response_format = {"type": "json_schema", "name": "City", "schema": {"type": "object"}}

    provider = GroqProvider(api_key="test-api-key")
    result = await provider.aresponses("openai/gpt-oss-20b", "hi", response_format=response_format)

    client.responses.parse.assert_not_called()
    assert client.responses.create.call_args.kwargs["text"] == {"format": response_format}
    # A raw dict response_format is passed through unparsed.
    assert not isinstance(result, ParsedResponse)


@pytest.mark.asyncio
@patch("any_llm.providers.groq.groq.AsyncOpenAI")
async def test_groq_aresponses_without_response_format(mock_openai_class: Mock) -> None:
    from any_llm.providers.groq.groq import GroqProvider

    client = AsyncMock()
    client.responses.create = AsyncMock(return_value=_make_openai_response("{}"))
    client.responses.parse = AsyncMock()
    mock_openai_class.return_value = client

    provider = GroqProvider(api_key="test-api-key")
    await provider.aresponses("openai/gpt-oss-20b", "hi")

    client.responses.parse.assert_not_called()
    assert "text" not in client.responses.create.call_args.kwargs
