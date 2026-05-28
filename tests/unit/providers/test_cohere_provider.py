from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.cohere.utils import (
    _convert_cohere_embedding_response,
    _convert_response,
    _create_openai_chunk_from_cohere_chunk,
    _patch_messages,
)
from any_llm.types.completion import CompletionParams, CreateEmbeddingResponse, ReasoningEffort


def _mk_provider() -> Any:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    return CohereProvider(api_key="test-api-key")


def test_preprocess_response_format() -> None:
    provider = _mk_provider()

    class StructuredOutput(BaseModel):
        foo: str
        bar: int

    json_schema = {"type": "json_object", "schema": StructuredOutput.model_json_schema()}

    outp_basemodel = provider._preprocess_response_format(StructuredOutput)

    outp_dict = provider._preprocess_response_format(json_schema)

    assert isinstance(outp_basemodel, dict)
    assert isinstance(outp_dict, dict)

    assert outp_basemodel == outp_dict


@pytest.mark.asyncio
async def test_stream_and_response_format_combination_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_parallel_tool_calls_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                parallel_tool_calls=False,
            )
        )


def test_patch_messages_removes_name_from_tool_messages() -> None:
    """Test that _patch_messages removes 'name' field from tool messages."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "I'll check the weather for you.",
            "tool_calls": [{"id": "call_123", "function": {"name": "get_weather"}}],
        },
        {"role": "tool", "name": "get_weather", "content": "It's sunny", "tool_call_id": "call_123"},
        {"role": "assistant", "content": "The weather is sunny."},
    ]

    result = _patch_messages(messages)

    # Check that the tool message no longer has 'name' field
    tool_message = next(msg for msg in result if msg["role"] == "tool")
    assert "name" not in tool_message
    assert tool_message["content"] == "It's sunny"
    assert tool_message["tool_call_id"] == "call_123"

    # Check that other messages are unchanged
    user_message = next(msg for msg in result if msg["role"] == "user")
    assert user_message == {"role": "user", "content": "What's the weather?"}


def test_patch_messages_converts_assistant_content_to_tool_plan() -> None:
    """Test that _patch_messages converts assistant content to tool_plan when tool_calls present."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Calculate 2+2"},
        {
            "role": "assistant",
            "content": "I'll calculate that for you.",
            "tool_calls": [{"id": "call_456", "function": {"name": "calculator"}}],
        },
        {"role": "tool", "content": "4", "tool_call_id": "call_456"},
    ]

    result = _patch_messages(messages)

    # Check that assistant message with tool_calls has content moved to tool_plan
    assistant_message = next(msg for msg in result if msg["role"] == "assistant" and msg.get("tool_calls"))
    assert "content" not in assistant_message
    assert assistant_message["tool_plan"] == "I'll calculate that for you."
    assert assistant_message["tool_calls"] == [{"id": "call_456", "function": {"name": "calculator"}}]


def test_patch_messages_leaves_regular_assistant_messages_unchanged() -> None:
    """Test that _patch_messages doesn't modify assistant messages without tool_calls."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
        {"role": "user", "content": "Thanks"},
    ]

    result = _patch_messages(messages)

    # Messages should be unchanged
    assert result == messages

    # Verify assistant message still has content
    assistant_message = next(msg for msg in result if msg["role"] == "assistant")
    assert assistant_message["content"] == "Hello! How can I help you?"
    assert "tool_plan" not in assistant_message


def test_patch_messages_with_invalid_tool_sequence_raises_error() -> None:
    """Test that an invalid tool message sequence raises a ValueError."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "tool", "name": "get_weather", "content": "It's sunny", "tool_call_id": "call_123"},
    ]
    with pytest.raises(ValueError, match=r"A tool message must be preceded by an assistant message with tool_calls."):
        _patch_messages(messages)


def test_patch_messages_with_parallel_tool_messages() -> None:
    """Test that multiple consecutive tool messages after an assistant message are valid."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Get weather for Paris and London."},
        {
            "role": "assistant",
            "content": "I'll check both cities.",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'}},
                {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"location":"London"}'}},
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "Sunny in Paris", "tool_call_id": "call_1"},
        {"role": "tool", "name": "get_weather", "content": "Rainy in London", "tool_call_id": "call_2"},
    ]

    result = _patch_messages(messages)

    tool_messages = [msg for msg in result if msg["role"] == "tool"]
    assert len(tool_messages) == 2
    for msg in tool_messages:
        assert "name" not in msg


def test_patch_messages_tool_after_tool_without_assistant_raises() -> None:
    """Test that a tool message group not preceded by an assistant message raises."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "result1", "tool_call_id": "call_1"},
        {"role": "tool", "content": "result2", "tool_call_id": "call_2"},
    ]
    with pytest.raises(ValueError, match=r"A tool message must be preceded by an assistant message with tool_calls."):
        _patch_messages(messages)


def test_preprocess_response_format_dataclass() -> None:
    from dataclasses import dataclass

    provider = _mk_provider()

    @dataclass
    class StructuredOutputDC:
        foo: str
        bar: int

    result = provider._preprocess_response_format(StructuredOutputDC)

    assert isinstance(result, dict)
    assert result["type"] == "json_object"
    assert "properties" in result["schema"]
    assert "foo" in result["schema"]["properties"]
    assert "bar" in result["schema"]["properties"]


def test_preprocess_response_format_unsupported_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(ValueError, match="Unsupported response_format"):
        provider._preprocess_response_format("invalid")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reasoning_effort", "expected_thinking"),
    [
        ("auto", None),
        ("none", {"type": "disabled"}),
        (None, {"type": "disabled"}),
        ("minimal", {"type": "enabled", "token_budget": 256}),
        ("low", {"type": "enabled", "token_budget": 1024}),
        ("medium", {"type": "enabled", "token_budget": 8192}),
        ("high", {"type": "enabled", "token_budget": 24576}),
        ("xhigh", {"type": "enabled", "token_budget": 32768}),
    ],
)
async def test_reasoning_effort_mapped_to_thinking(
    reasoning_effort: ReasoningEffort | None, expected_thinking: dict[str, Any] | None
) -> None:
    """Test reasoning_effort maps to Cohere thinking config like Gemini."""
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    with patch("any_llm.providers.cohere.cohere.cohere") as mock_cohere:
        mock_client = Mock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        mock_client.chat = AsyncMock(return_value=Mock())

        with patch("any_llm.providers.cohere.cohere._convert_response", return_value=Mock()):
            provider = CohereProvider(api_key="test-api-key")
            await provider._acompletion(
                CompletionParams(
                    model_id="command-r-plus",
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort=reasoning_effort,
                ),
            )

            call_kwargs = mock_client.chat.call_args[1]
            assert "reasoning_effort" not in call_kwargs
            if expected_thinking is None:
                assert "thinking" not in call_kwargs
            else:
                assert call_kwargs["thinking"] == expected_thinking


def _mock_tool_call(tool_id: str, name: str, arguments: str) -> Mock:
    """Create a mock Cohere ToolCallV2."""
    tc = Mock()
    tc.id = tool_id
    tc.function = Mock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _mock_v2chat_response(tool_calls: list[Mock], tool_plan: str = "I'll help.") -> Mock:
    """Create a mock V2ChatResponse with multiple tool calls."""
    response = Mock()
    response.finish_reason = "TOOL_CALL"
    response.id = "resp-123"
    response.created = 0
    response.message = Mock()
    response.message.tool_calls = tool_calls
    response.message.tool_plan = tool_plan
    response.usage = Mock()
    response.usage.tokens = Mock()
    response.usage.tokens.input_tokens = 10
    response.usage.tokens.output_tokens = 20
    return response


def test_convert_response_multiple_tool_calls() -> None:
    """Non-streaming: all tool calls are converted, not just the first."""
    tc1 = _mock_tool_call("call_1", "get_weather", '{"city":"NYC"}')
    tc2 = _mock_tool_call("call_2", "get_time", '{"tz":"EST"}')
    tc3 = _mock_tool_call("call_3", "get_news", '{"topic":"tech"}')

    result = _convert_response(_mock_v2chat_response([tc1, tc2, tc3]), model="command-r-plus")

    assert len(result.choices) == 1
    assert result.choices[0].finish_reason == "tool_calls"
    tool_calls = result.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 3
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].function.name == "get_weather"  # type: ignore[union-attr]
    assert tool_calls[0].function.arguments == '{"city":"NYC"}'  # type: ignore[union-attr]
    assert tool_calls[1].id == "call_2"
    assert tool_calls[1].function.name == "get_time"  # type: ignore[union-attr]
    assert tool_calls[2].id == "call_3"
    assert tool_calls[2].function.name == "get_news"  # type: ignore[union-attr]


def test_convert_response_single_tool_call() -> None:
    """Non-streaming: single tool call still works correctly."""
    tc = _mock_tool_call("call_1", "get_weather", '{"city":"NYC"}')

    result = _convert_response(_mock_v2chat_response([tc]), model="command-r-plus")

    assert len(result.choices[0].message.tool_calls) == 1  # type: ignore[arg-type]
    assert result.choices[0].message.tool_calls[0].id == "call_1"  # type: ignore[index]


def test_streaming_tool_call_start_uses_chunk_index() -> None:
    """Streaming: tool-call-start uses chunk.index, not hardcoded 0."""
    chunk = Mock()
    chunk.type = "tool-call-start"
    chunk.index = 2
    chunk.delta = Mock()
    chunk.delta.message = Mock()
    chunk.delta.message.tool_calls = Mock()
    chunk.delta.message.tool_calls.id = "call_abc"
    chunk.delta.message.tool_calls.function = Mock()
    chunk.delta.message.tool_calls.function.name = "get_weather"

    result = _create_openai_chunk_from_cohere_chunk(chunk)

    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 2
    assert tool_calls[0].id == "call_abc"
    assert tool_calls[0].function.name == "get_weather"  # type: ignore[union-attr]


def test_streaming_tool_call_delta_uses_chunk_index() -> None:
    """Streaming: tool-call-delta uses chunk.index, not hardcoded 0."""
    chunk = Mock()
    chunk.type = "tool-call-delta"
    chunk.index = 3
    chunk.delta = Mock()
    chunk.delta.message = Mock()
    chunk.delta.message.tool_calls = Mock()
    chunk.delta.message.tool_calls.function = Mock()
    chunk.delta.message.tool_calls.function.arguments = '{"partial'

    result = _create_openai_chunk_from_cohere_chunk(chunk)

    tool_calls = result.choices[0].delta.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].index == 3
    assert tool_calls[0].function.arguments == '{"partial'  # type: ignore[union-attr]


def test_streaming_tool_call_index_defaults_to_zero_when_none() -> None:
    """Streaming: falls back to index 0 when chunk.index is None."""
    chunk = Mock()
    chunk.type = "tool-call-start"
    chunk.index = None
    chunk.delta = Mock()
    chunk.delta.message = Mock()
    chunk.delta.message.tool_calls = Mock()
    chunk.delta.message.tool_calls.id = "call_xyz"
    chunk.delta.message.tool_calls.function = Mock()
    chunk.delta.message.tool_calls.function.name = "search"

    result = _create_openai_chunk_from_cohere_chunk(chunk)

    assert result.choices[0].delta.tool_calls[0].index == 0  # type: ignore[index]


def _mock_embed_by_type_response(
    vectors: list[list[float]] | None = None,
    *,
    int8_vectors: list[list[int]] | None = None,
    input_tokens: int = 10,
    response_id: str = "emb-123",
) -> Mock:
    """Create a mock Cohere EmbedByTypeResponse."""
    response = Mock()
    response.id = response_id
    response.embeddings = Mock(spec=[])
    response.embeddings.float_ = vectors
    response.embeddings.int8 = int8_vectors
    response.embeddings.uint8 = None
    response.embeddings.binary = None
    response.embeddings.ubinary = None
    response.meta = Mock()
    response.meta.tokens = Mock()
    response.meta.tokens.input_tokens = input_tokens
    return response


def test_convert_cohere_embedding_response_single_vector() -> None:
    vectors = [[0.1, 0.2, 0.3]]
    mock_response = _mock_embed_by_type_response(vectors, input_tokens=5)

    result = _convert_cohere_embedding_response("embed-v4.0", mock_response)

    assert isinstance(result, CreateEmbeddingResponse)
    assert result.model == "embed-v4.0"
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.data[0].index == 0
    assert result.data[0].object == "embedding"
    assert result.usage.prompt_tokens == 5
    assert result.usage.total_tokens == 5
    assert result.object == "list"


def test_convert_cohere_embedding_response_multiple_vectors() -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    mock_response = _mock_embed_by_type_response(vectors, input_tokens=15)

    result = _convert_cohere_embedding_response("embed-v4.0", mock_response)

    assert result.model == "embed-v4.0"
    assert len(result.data) == 3
    for i, embedding in enumerate(result.data):
        assert embedding.index == i
        assert embedding.embedding == vectors[i]
    assert result.usage.prompt_tokens == 15


def test_convert_cohere_embedding_response_no_meta() -> None:
    vectors = [[0.1, 0.2]]
    mock_response = _mock_embed_by_type_response(vectors)
    mock_response.meta = None

    result = _convert_cohere_embedding_response("embed-v4.0", mock_response)

    assert result.model == "embed-v4.0"
    assert result.usage.prompt_tokens == 0
    assert result.usage.total_tokens == 0
    assert len(result.data) == 1


def test_convert_cohere_embedding_response_empty_vectors() -> None:
    mock_response = _mock_embed_by_type_response(vectors=[])

    result = _convert_cohere_embedding_response("embed-v4.0", mock_response)

    assert len(result.data) == 0
    assert result.object == "list"


def test_convert_cohere_embedding_response_falls_back_to_int8() -> None:
    """When float_ is empty but int8 has data, int8 vectors are returned as floats."""
    mock_response = _mock_embed_by_type_response(vectors=None, int8_vectors=[[1, 2, 3], [4, 5, 6]])

    result = _convert_cohere_embedding_response("embed-v4.0", mock_response)

    assert len(result.data) == 2
    assert result.data[0].embedding == [1.0, 2.0, 3.0]
    assert result.data[1].embedding == [4.0, 5.0, 6.0]


def test_convert_embedding_params_single_string() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    result = CohereProvider._convert_embedding_params("hello world")

    assert result["texts"] == ["hello world"]
    assert result["input_type"] == "search_document"
    assert result["embedding_types"] == ["float"]


def test_convert_embedding_params_list_of_strings() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    result = CohereProvider._convert_embedding_params(["hello", "world"])

    assert result["texts"] == ["hello", "world"]
    assert result["input_type"] == "search_document"


def test_convert_embedding_params_custom_input_type() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    result = CohereProvider._convert_embedding_params("query text", input_type="search_query")

    assert result["texts"] == ["query text"]
    assert result["input_type"] == "search_query"


def test_convert_embedding_params_custom_embedding_types() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    result = CohereProvider._convert_embedding_params("text", embedding_types=["float", "int8"])

    assert result["embedding_types"] == ["float", "int8"]


def test_convert_embedding_params_extra_kwargs() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    result = CohereProvider._convert_embedding_params("text", truncate="END")

    assert result["truncate"] == "END"
    assert result["texts"] == ["text"]


@pytest.mark.asyncio
async def test_aembedding_calls_client() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    mock_response = _mock_embed_by_type_response([[0.1, 0.2, 0.3]])

    with patch("any_llm.providers.cohere.cohere.cohere") as mock_cohere:
        mock_client = Mock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        mock_client.embed = AsyncMock(return_value=mock_response)

        provider = CohereProvider(api_key="test-key")
        result = await provider._aembedding("embed-v4.0", "hello world")

        assert isinstance(result, CreateEmbeddingResponse)
        assert result.model == "embed-v4.0"
        assert len(result.data) == 1
        mock_client.embed.assert_called_once()
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["model"] == "embed-v4.0"
        assert call_kwargs["texts"] == ["hello world"]
        assert call_kwargs["input_type"] == "search_document"


@pytest.mark.asyncio
async def test_aembedding_passes_custom_input_type() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    mock_response = _mock_embed_by_type_response([[0.1, 0.2]])

    with patch("any_llm.providers.cohere.cohere.cohere") as mock_cohere:
        mock_client = Mock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        mock_client.embed = AsyncMock(return_value=mock_response)

        provider = CohereProvider(api_key="test-key")
        await provider._aembedding("embed-v4.0", ["query"], input_type="search_query")

        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "search_query"
        assert call_kwargs["texts"] == ["query"]


def test_patch_messages_preserves_multimodal_user_content() -> None:
    """Verify _patch_messages passes through user messages with image content blocks."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        },
    ]

    result = _patch_messages(messages)

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2
    assert result[0]["content"][0] == {"type": "text", "text": "What is in this image?"}
    assert result[0]["content"][1] == {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}


def test_patch_messages_preserves_base64_image_content() -> None:
    """Verify _patch_messages preserves data URI image content blocks."""
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANS"
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        },
    ]

    result = _patch_messages(messages)

    assert result[0]["content"][1]["image_url"]["url"] == data_uri


def test_patch_messages_multimodal_with_tool_calls() -> None:
    """Image content in earlier user messages is preserved alongside tool call flows."""
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ],
        },
        {
            "role": "assistant",
            "content": "Let me look that up.",
            "tool_calls": [{"id": "call_1", "function": {"name": "analyze_image"}}],
        },
        {"role": "tool", "name": "analyze_image", "content": "A cat", "tool_call_id": "call_1"},
        {"role": "assistant", "content": "It's a cat."},
    ]

    result = _patch_messages(messages)

    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    assistant_with_tools = result[1]
    assert "content" not in assistant_with_tools
    assert assistant_with_tools["tool_plan"] == "Let me look that up."

    tool_msg = result[2]
    assert "name" not in tool_msg


@pytest.mark.asyncio
async def test_completion_with_image_content() -> None:
    """End-to-end: image content blocks are forwarded to the Cohere chat API."""
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    mock_response = Mock()
    mock_response.id = "resp-img"
    mock_response.created = 0
    mock_response.finish_reason = "COMPLETE"
    mock_response.message = Mock()
    mock_response.message.tool_calls = None
    mock_response.message.tool_plan = None
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "A landscape photo."
    mock_response.message.content = [text_block]
    mock_response.usage = Mock()
    mock_response.usage.tokens = Mock()
    mock_response.usage.tokens.input_tokens = 50
    mock_response.usage.tokens.output_tokens = 10

    with patch("any_llm.providers.cohere.cohere.cohere") as mock_cohere:
        mock_client = Mock()
        mock_cohere.AsyncClientV2.return_value = mock_client
        mock_client.chat = AsyncMock(return_value=mock_response)

        provider = CohereProvider(api_key="test-key")
        result = await provider._acompletion(
            CompletionParams(
                model_id="command-a-03-2025",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
                        ],
                    }
                ],
            ),
        )

        assert result.choices[0].message.content == "A landscape photo."  # type: ignore[union-attr]

        call_kwargs = mock_client.chat.call_args
        sent_messages = call_kwargs[1]["messages"]
        user_msg = sent_messages[0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][1]["type"] == "image_url"
