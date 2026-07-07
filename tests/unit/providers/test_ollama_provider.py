from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from ollama import ChatResponse as OllamaChatResponse
from ollama import Message as OllamaMessage

from any_llm.providers.ollama.ollama import OllamaProvider
from any_llm.providers.ollama.utils import _create_chat_completion_from_ollama_response
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
async def test_create_chat_completion_extracts_think_content() -> None:
    """Test that <think> content is correctly extracted into reasoning field."""
    # Create a mock Ollama response with <think> tags in content
    mock_message = Mock(spec=OllamaMessage)
    mock_message.content = "<think>This is my reasoning process</think>This is the actual response"
    mock_message.thinking = None
    mock_message.tool_calls = None
    mock_message.role = "assistant"

    mock_response = Mock(spec=OllamaChatResponse)
    mock_response.message = mock_message
    mock_response.created_at = "2024-01-01T12:00:00.000000Z"
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.model = "llama3.1"
    mock_response.done_reason = "stop"

    result = _create_chat_completion_from_ollama_response(mock_response)

    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "This is my reasoning process"

    assert result.choices[0].message.content == "This is the actual response"


@pytest.mark.asyncio
async def test_completion_with_dataclass_response_format() -> None:
    """Test that dataclass response_format is converted to JSON schema."""
    from dataclasses import dataclass

    @dataclass
    class TestOutput:
        name: str
        value: int

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="llama3.1",
                    messages=[{"role": "user", "content": "Hello"}],
                    response_format=TestOutput,
                ),
            )

            call_kwargs = provider.client.chat.call_args[1]
            assert isinstance(call_kwargs["format"], dict)
            assert "properties" in call_kwargs["format"]
            assert "name" in call_kwargs["format"]["properties"]
            assert "value" in call_kwargs["format"]["properties"]


@pytest.mark.asyncio
async def test_completion_with_dict_response_format() -> None:
    """Test that dict response_format is passed through unchanged."""
    response_format = {"type": "json_object"}

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="llama3.1",
                    messages=[{"role": "user", "content": "Hello"}],
                    response_format=response_format,
                ),
            )

            call_kwargs = provider.client.chat.call_args[1]
            assert call_kwargs["format"] == response_format


@pytest.mark.asyncio
async def test_streaming_completion_passes_format_top_level() -> None:
    """Test that response_format is converted and passed as a top-level `format` kwarg when streaming."""
    from dataclasses import dataclass

    @dataclass
    class TestOutput:
        name: str

    async def empty_async_iter() -> AsyncIterator[None]:
        return
        yield

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=empty_async_iter())

        result = await provider._acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
                stream=True,
            ),
        )
        async for _ in result:  # type: ignore[union-attr]
            pass

        call_kwargs = provider.client.chat.call_args[1]
        assert isinstance(call_kwargs["format"], dict)
        assert "name" in call_kwargs["format"]["properties"]
        assert "format" not in call_kwargs.get("options", {})


@pytest.mark.asyncio
async def test_streaming_completion_passes_tools_top_level() -> None:
    """Test that tools are passed as a top-level kwarg to client.chat(), not inside options."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]

    async def empty_async_iter() -> AsyncIterator[None]:
        return
        yield

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=empty_async_iter())

        result = await provider._acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=tools,
                stream=True,
            ),
        )
        async for _ in result:  # type: ignore[union-attr]
            pass

        call_kwargs = provider.client.chat.call_args[1]
        assert call_kwargs["tools"] == tools
        assert "tools" not in call_kwargs.get("options", {})


@pytest.mark.asyncio
async def test_streaming_assigns_distinct_tool_call_indices() -> None:
    """Ollama streams each tool call in its own chunk and the chunk converter
    stamps every tool-call delta with index=0. When a model emits several tool
    calls, a consumer that accumulates streaming deltas by index concatenates
    the arguments of distinct calls into one invalid JSON string. The provider
    must reassign a stream-global, monotonically increasing index so each tool
    call stays in its own slot."""

    def _make_chunk(name: str, arguments: dict[str, str]) -> Mock:
        func = Mock()
        func.name = name
        func.arguments = arguments
        tool_call = Mock()
        tool_call.function = func

        message = Mock(spec=OllamaMessage)
        message.content = None
        message.role = "assistant"
        message.thinking = None
        message.tool_calls = [tool_call]

        chunk = Mock(spec=OllamaChatResponse)
        chunk.message = message
        chunk.created_at = None
        chunk.model = "llama3.1"
        chunk.done_reason = None
        chunk.prompt_eval_count = None
        chunk.eval_count = None
        return chunk

    chunks = [
        _make_chunk("find", {"query": "diode connected nmos"}),
        _make_chunk("find", {"query": "5t nmos ota"}),
        _make_chunk("find", {"query": "source follower"}),
    ]

    async def chunk_iter() -> AsyncIterator[Mock]:
        for chunk in chunks:
            yield chunk

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=chunk_iter())

        result = await provider._acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "find three fixtures"}],
                stream=True,
            ),
        )

        indices = []
        async for chunk in result:  # type: ignore[union-attr]
            for choice in chunk.choices:
                for tool_call in choice.delta.tool_calls or []:
                    indices.append(tool_call.index)

    assert indices == [0, 1, 2], f"expected distinct per-call indices, got {indices}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_think"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "high"),
    ],
)
async def test_reasoning_effort_maps_to_think_level(reasoning_effort: str, expected_think: str) -> None:
    """Test that reasoning_effort is mapped to Ollama's leveled `think` parameter."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="qwen3",
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
                ),
            )

            call_kwargs = provider.client.chat.call_args[1]
            assert call_kwargs["think"] == expected_think
            assert "reasoning_effort" not in call_kwargs.get("options", {})
            assert "think" not in call_kwargs.get("options", {})


@pytest.mark.asyncio
async def test_reasoning_effort_none_disables_think() -> None:
    """Test that reasoning_effort 'none' explicitly disables thinking."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="qwen3",
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort="none",
                ),
            )

            call_kwargs = provider.client.chat.call_args[1]
            assert call_kwargs["think"] is False
            assert "reasoning_effort" not in call_kwargs.get("options", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", None])
async def test_reasoning_effort_auto_leaves_think_unset(reasoning_effort: str | None) -> None:
    """Test that 'auto'/None leave `think` unset so Ollama uses its per-model default."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="qwen3",
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
                ),
            )

            call_kwargs = provider.client.chat.call_args[1]
            assert call_kwargs["think"] is None
            assert "reasoning_effort" not in call_kwargs.get("options", {})


@pytest.mark.asyncio
async def test_streaming_completion_passes_think_level_top_level() -> None:
    """Test that the leveled `think` is passed as a top-level kwarg to client.chat(), not in options."""

    async def empty_async_iter() -> AsyncIterator[None]:
        return
        yield

    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=empty_async_iter())

        result = await provider._acompletion(
            CompletionParams(
                model_id="qwen3",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort="medium",
                stream=True,
            ),
        )
        async for _ in result:  # type: ignore[union-attr]
            pass

        call_kwargs = provider.client.chat.call_args[1]
        assert call_kwargs["think"] == "medium"
        assert "think" not in call_kwargs.get("options", {})
        assert "reasoning_effort" not in call_kwargs.get("options", {})
