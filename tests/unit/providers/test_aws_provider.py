import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

from any_llm.providers.bedrock import BedrockProvider
from any_llm.providers.bedrock.utils import _create_openai_chunk_from_aws_chunk
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_aws_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.bedrock.bedrock._convert_response"),
        patch("boto3.Session"),
        patch("boto3.client") as mock_boto3_client,
    ):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        mock_client.converse.return_value = {"output": {"message": {"content": [{"text": "response"}]}}}
        yield mock_boto3_client


def test_boto3_client_created_with_api_base() -> None:
    """Test that boto3.client is created with api_base as endpoint_url when provided."""
    custom_endpoint = "https://custom-bedrock-endpoint.amazonaws.com"

    with mock_aws_provider() as mock_boto3_client:
        provider = BedrockProvider(api_base=custom_endpoint, api_key="test_key")
        provider._completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=custom_endpoint)


def test_boto3_client_created_without_api_base() -> None:
    """Test that boto3.client is created with None endpoint_url when api_base is not provided."""

    with mock_aws_provider() as mock_boto3_client:
        provider = BedrockProvider(api_key="test_key")
        provider._completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None)


def test_completion_with_kwargs() -> None:
    """Test that additional kwargs are passed correctly to converse method."""
    model_id = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_aws_provider() as mock_boto3_client:
        provider = BedrockProvider(api_key="test_key")
        provider._completion(
            CompletionParams(
                model_id=model_id,
                messages=messages,
                max_tokens=100,
            ),
            guardrailConfig={
                "guardrailIdentifier": "Guardrail ID",
                "guardrailVersion": "Guardrail version",
                "trace": "enabled",
            },
        )

        mock_boto3_client.return_value.converse.assert_called_once_with(
            guardrailConfig={
                "guardrailIdentifier": "Guardrail ID",
                "guardrailVersion": "Guardrail version",
                "trace": "enabled",
            },
            inferenceConfig={
                "maxTokens": 100,
            },
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            modelId=model_id,
        )


@contextmanager
def mock_aws_embedding_provider():  # type: ignore[no-untyped-def]
    """Mock AWS provider specifically for embedding tests."""
    with (
        patch("boto3.Session"),
        patch("boto3.client") as mock_boto3_client,
    ):
        mock_client = Mock()
        mock_boto3_client.return_value = mock_client
        yield mock_boto3_client, mock_client


def test_embedding_single_string() -> None:
    """Test embedding with a single string input."""
    model_id = "amazon.titan-embed-text-v1"
    input_text = "Hello world"

    mock_response_body = {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}

    with mock_aws_embedding_provider() as (mock_boto3_client, mock_client):
        mock_client.invoke_model.return_value = {"body": Mock(read=Mock(return_value=json.dumps(mock_response_body)))}

        provider = BedrockProvider(api_key="test_key")
        response = provider._embedding(model_id, input_text)

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None)

        expected_request_body = {"inputText": input_text}
        mock_client.invoke_model.assert_called_once_with(modelId=model_id, body=json.dumps(expected_request_body))

        assert response.model == model_id
        assert response.object == "list"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.data[0].index == 0
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5


def test_embedding_list_of_strings() -> None:
    """Test embedding with a list of strings."""
    model_id = "amazon.titan-embed-text-v1"
    input_texts = ["Hello world", "Goodbye world"]

    mock_response_bodies = [
        {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5},
        {"embedding": [0.4, 0.5, 0.6], "inputTextTokenCount": 6},
    ]

    with mock_aws_embedding_provider() as (mock_boto3_client, mock_client):
        mock_client.invoke_model.side_effect = [
            {"body": Mock(read=Mock(return_value=json.dumps(mock_response_bodies[0])))},
            {"body": Mock(read=Mock(return_value=json.dumps(mock_response_bodies[1])))},
        ]

        provider = BedrockProvider(api_key="test_key")
        response = provider._embedding(model_id, input_texts)

        mock_boto3_client.assert_called_once_with("bedrock-runtime", endpoint_url=None)

        assert mock_client.invoke_model.call_count == 2
        expected_calls = [({"inputText": "Hello world"}, model_id), ({"inputText": "Goodbye world"}, model_id)]
        for i, (expected_body, expected_model) in enumerate(expected_calls):
            actual_call = mock_client.invoke_model.call_args_list[i]
            assert actual_call[1]["modelId"] == expected_model
            assert json.loads(actual_call[1]["body"]) == expected_body

        assert response.model == model_id
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.data[0].index == 0
        assert response.data[1].embedding == [0.4, 0.5, 0.6]
        assert response.data[1].index == 1
        assert response.usage.prompt_tokens == 11
        assert response.usage.total_tokens == 11


def test_streaming_chunk_with_tool_use_start() -> None:
    """Test streaming chunk with tool use in contentBlockStart."""
    chunk = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {
                "toolUse": {
                    "toolUseId": "tool-123",
                    "name": "get_weather",
                }
            },
        }
    }
    tool_index_map: dict[int, int] = {}
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model", tool_index_map)

    assert result is not None
    assert len(result.choices) == 1
    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "tool-123"
    assert tool_call.function is not None
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == ""
    assert tool_index_map[0] == 0


def test_streaming_chunk_with_tool_use_delta() -> None:
    """Test streaming chunk with tool use in contentBlockDelta."""
    tool_index_map: dict[int, int] = {0: 0}
    chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {
                "toolUse": {
                    "input": '{"location": "Paris"}',
                }
            },
        }
    }
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model", tool_index_map)

    assert result is not None
    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.function is not None
    assert tool_call.function.arguments == '{"location": "Paris"}'


def test_streaming_chunk_with_multiple_tool_calls() -> None:
    """Test streaming with multiple tool calls tracks indices correctly."""
    tool_index_map: dict[int, int] = {}

    chunk1 = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"toolUseId": "tool-1", "name": "func_a"}},
        }
    }
    result1 = _create_openai_chunk_from_aws_chunk(chunk1, "test-model", tool_index_map)
    assert result1 is not None
    assert result1.choices[0].delta.tool_calls is not None
    assert result1.choices[0].delta.tool_calls[0].index == 0

    chunk2 = {
        "contentBlockStart": {
            "contentBlockIndex": 1,
            "start": {"toolUse": {"toolUseId": "tool-2", "name": "func_b"}},
        }
    }
    result2 = _create_openai_chunk_from_aws_chunk(chunk2, "test-model", tool_index_map)
    assert result2 is not None
    assert result2.choices[0].delta.tool_calls is not None
    assert result2.choices[0].delta.tool_calls[0].index == 1

    assert tool_index_map == {0: 0, 1: 1}


def test_streaming_chunk_with_reasoning_content_start() -> None:
    """Test streaming chunk with reasoning content in contentBlockStart."""
    chunk = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"reasoningContent": {}},
        }
    }
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == ""


def test_streaming_chunk_with_reasoning_content_delta() -> None:
    """Test streaming chunk with reasoning content in contentBlockDelta."""
    chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"reasoningContent": {"text": "Let me think..."}},
        }
    }
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Let me think..."


def test_streaming_chunk_with_text_content() -> None:
    """Test streaming chunk with text content."""
    chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"text": "Hello world"},
        }
    }
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].delta.content == "Hello world"


def test_streaming_chunk_message_stop_tool_use() -> None:
    """Test streaming chunk with messageStop for tool_use."""
    chunk = {"messageStop": {"stopReason": "tool_use"}}
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].finish_reason == "tool_calls"


def test_streaming_chunk_message_stop_max_tokens() -> None:
    """Test streaming chunk with messageStop for max_tokens."""
    chunk = {"messageStop": {"stopReason": "max_tokens"}}
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].finish_reason == "length"


def test_streaming_chunk_message_stop_end_turn() -> None:
    """Test streaming chunk with messageStop for end_turn."""
    chunk = {"messageStop": {"stopReason": "end_turn"}}
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].finish_reason == "stop"


def test_streaming_chunk_message_start() -> None:
    """Test streaming chunk with messageStart."""
    chunk = {"messageStart": {"role": "assistant"}}
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].delta.content == ""


def test_streaming_chunk_unknown_type_returns_none() -> None:
    """Test streaming chunk with unknown type returns None."""
    chunk: dict[str, Any] = {"unknownField": {}}
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is None


def test_streaming_chunk_content_block_start_text() -> None:
    """Test streaming chunk with contentBlockStart for text (no special block)."""
    chunk = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {},
        }
    }
    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.choices[0].delta.content == ""
