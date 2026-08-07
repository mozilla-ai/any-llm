import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, get_args
from unittest.mock import Mock, patch

import botocore.session
import pytest
from botocore.tokens import ScopedEnvTokenProvider

from any_llm.exceptions import InvalidRequestError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.providers.bedrock.utils import (
    REASONING_EFFORT_TO_THINKING_BUDGETS,
    _convert_images_for_bedrock,
    _convert_messages,
    _convert_response,
    _convert_tool_spec,
    _create_openai_chunk_from_aws_chunk,
)
from any_llm.types.completion import CompletionParams, ReasoningEffort


@contextmanager
def mock_aws_provider():  # type: ignore[no-untyped-def]
    """Mock boto3.Session so the provider builds its runtime client from a mocked session.

    Yields the mocked ``session.client`` callable (the equivalent of the old bare
    ``boto3.client`` mock), since BedrockProvider now builds a dedicated ``boto3.Session``
    per instance instead of calling the module-level ``boto3.client`` directly.
    """
    with (
        patch("any_llm.providers.bedrock.bedrock._convert_response"),
        patch("boto3.Session") as mock_session_cls,
    ):
        mock_client = Mock()
        mock_session_cls.return_value.client.return_value = mock_client
        mock_client.converse.return_value = {"output": {"message": {"content": [{"text": "response"}]}}}
        yield mock_session_cls.return_value.client


def test_boto3_client_created_with_api_base() -> None:
    """Test that the session's client is created with api_base as endpoint_url when provided."""
    custom_endpoint = "https://custom-bedrock-endpoint.amazonaws.com"

    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider(api_base=custom_endpoint, api_key="test_key")
        provider._completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_client_call.assert_called_once()
        call_args, call_kwargs = mock_client_call.call_args
        assert call_args == ("bedrock-runtime",)
        assert call_kwargs["endpoint_url"] == custom_endpoint
        # api_key was provided, so the client is configured to sign with the bearer token.
        assert call_kwargs["config"].signature_version == "bearer"


def test_boto3_client_created_without_api_base() -> None:
    """Test that the session's client is created with None endpoint_url when api_base is not provided."""

    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider(api_key="test_key")
        provider._completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_client_call.assert_called_once()
        call_args, call_kwargs = mock_client_call.call_args
        assert call_args == ("bedrock-runtime",)
        assert call_kwargs["endpoint_url"] is None
        assert call_kwargs["config"].signature_version == "bearer"


def test_api_key_registers_scoped_bearer_token_provider() -> None:
    """Test that api_key is forwarded as a bearer token via a per-instance token provider.

    This is what actually makes api_key/AWS_BEARER_TOKEN_BEDROCK usable per-request/per-tenant,
    instead of requiring a process-wide env var mutation.
    """
    with mock_aws_provider():
        provider = BedrockProvider(api_key="my-bearer-token")

    mock_session = provider._boto_session
    assert mock_session is not None
    mock_session._session.register_component.assert_called_once()
    component_name, token_provider = mock_session._session.register_component.call_args[0]
    assert component_name == "token_provider"
    assert token_provider.environ == {"AWS_BEARER_TOKEN_BEDROCK": "my-bearer-token"}


def test_scoped_token_provider_resolves_bearer_token_via_real_botocore_session() -> None:
    """Test the registered token provider against the real botocore session contract.

    Goes one level deeper than asserting `register_component` was called with the right
    arguments: registers the provider on a real `botocore.session.Session` (no boto3/botocore
    mocking at all) and confirms botocore's own `get_auth_token(signing_name="bedrock")` actually
    resolves our token through it, exactly as it would when a real client is constructed.
    """
    real_session = botocore.session.Session()  # type: ignore[no-untyped-call]
    real_session.register_component(  # type: ignore[no-untyped-call]
        "token_provider",
        ScopedEnvTokenProvider(  # type: ignore[no-untyped-call]
            real_session, environ={"AWS_BEARER_TOKEN_BEDROCK": "my-bearer-token"}
        ),
    )

    auth_token = real_session.get_auth_token(signing_name="bedrock")  # type: ignore[no-untyped-call]

    assert auth_token is not None
    assert auth_token.token == "my-bearer-token"  # noqa: S105


def test_no_api_key_skips_bearer_token_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that no token provider is registered and no bearer config is forced without api_key."""
    # Guard against a real AWS_BEARER_TOKEN_BEDROCK in the ambient environment, which would make
    # the provider resolve a bearer token anyway and fail this test for unrelated reasons.
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider()
        provider._completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

    assert provider._boto_session is not None
    provider._boto_session._session.register_component.assert_not_called()
    assert "config" not in mock_client_call.call_args[1]


def test_completion_with_timeout_builds_distinct_client_with_timeout_config() -> None:
    """Test that a `timeout` kwarg is popped before reaching converse and applied via client config.

    boto3's Converse API has no `timeout` parameter, so it must never be forwarded to it.
    """
    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider(api_key="test_key")
        provider._completion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]),
            timeout=5,
        )

        # one call to build the base client, one to build the timeout-scoped client
        assert mock_client_call.call_count == 2
        timeout_call_kwargs = mock_client_call.call_args_list[1][1]
        assert timeout_call_kwargs["config"].connect_timeout == 5
        assert timeout_call_kwargs["config"].read_timeout == 5
        # bearer auth (from api_key) must still be honored on the timeout-scoped client
        assert timeout_call_kwargs["config"].signature_version == "bearer"

        converse_call_kwargs = mock_client_call.return_value.converse.call_args[1]
        assert "timeout" not in converse_call_kwargs


def test_completion_with_timeout_and_no_api_key_uses_plain_timeout_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the timeout-only branch (no bearer config to merge into) still works."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)

    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider()
        provider._completion(
            CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]),
            timeout=5,
        )

        timeout_call_kwargs = mock_client_call.call_args_list[1][1]
        assert timeout_call_kwargs["config"].connect_timeout == 5
        assert timeout_call_kwargs["config"].signature_version is None


def test_user_supplied_config_is_merged_with_bearer_signature_version() -> None:
    """A caller-supplied `config=` kwarg is preserved and merged with the forced bearer signature_version."""
    from botocore.config import Config

    user_config = Config(region_name="us-west-2")  # type: ignore[no-untyped-call]
    with mock_aws_provider() as mock_client_call:
        BedrockProvider(api_key="test_key", config=user_config)

    call_kwargs = mock_client_call.call_args[1]
    assert call_kwargs["config"].region_name == "us-west-2"
    assert call_kwargs["config"].signature_version == "bearer"


def test_completion_timeout_client_is_cached_per_timeout_value() -> None:
    """Test that repeated calls with the same timeout reuse a single cached client."""
    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider(api_key="test_key")
        params = CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}])

        provider._completion(params, timeout=5)
        provider._completion(params, timeout=5)

        # base client + one timeout-scoped client, reused across both calls
        assert mock_client_call.call_count == 2
        assert len(provider._timeout_clients) == 1


def test_timeout_client_cache_is_bounded() -> None:
    """Test that the timeout-client cache evicts old entries instead of growing without limit."""
    with mock_aws_provider():
        provider = BedrockProvider(api_key="test_key")

        for timeout in range(provider._MAX_TIMEOUT_CLIENTS + 5):
            provider._client_for_timeout(float(timeout))

        assert len(provider._timeout_clients) == provider._MAX_TIMEOUT_CLIENTS


def test_timeout_client_creation_is_thread_safe() -> None:
    """Test that concurrent calls with the same uncached timeout build exactly one client.

    `_completion` runs on executor threads via `_acompletion`; without locking the cache-miss
    path, concurrent callers could each build (and leak) their own client for the same timeout.
    """
    with mock_aws_provider() as mock_client_call:
        provider = BedrockProvider(api_key="test_key")

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(lambda _: provider._client_for_timeout(5.0), range(16)))

        assert len({id(client) for client in results}) == 1
        # base client (built at construction) + exactly one timeout-scoped client
        assert mock_client_call.call_count == 2


def test_completion_timeout_with_custom_client_logs_warning_and_is_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that `timeout` is dropped with a warning (not a crash) when a custom client is used.

    any-llm doesn't own a custom client's construction, so it can't safely reconfigure its
    connection timeouts.
    """
    custom_client = Mock()
    custom_client.converse.return_value = {"output": {"message": {"content": [{"text": "response"}]}}}

    with patch("any_llm.providers.bedrock.bedrock._convert_response"):
        provider = BedrockProvider(client=custom_client)
        with caplog.at_level(logging.WARNING, logger="any_llm"):
            provider._completion(
                CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]),
                timeout=5,
            )

    custom_client.converse.assert_called_once()
    assert "timeout" not in custom_client.converse.call_args[1]
    assert "timeout" in caplog.text.lower()


def test_custom_client_used_when_provided() -> None:
    """Test that a user-provided boto3 client is used instead of creating a new one."""
    custom_client = Mock()
    custom_client.converse.return_value = {"output": {"message": {"content": [{"text": "response"}]}}}

    with (
        patch("any_llm.providers.bedrock.bedrock._convert_response"),
        patch("boto3.client") as mock_boto3_client,
    ):
        provider = BedrockProvider(client=custom_client)

        provider._completion(CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}]))

        mock_boto3_client.assert_not_called()
        custom_client.converse.assert_called_once()


def test_custom_client_skips_credential_verification() -> None:
    """Test that credential verification is skipped when a custom client is provided."""
    custom_client = Mock()

    with (
        patch("boto3.Session") as mock_session,
        patch("boto3.client"),
    ):
        BedrockProvider(client=custom_client)

        mock_session.assert_not_called()


def test_custom_client_kwargs_not_forwarded_to_boto3() -> None:
    """Test that the 'client' kwarg is consumed and not forwarded to boto3.client."""
    custom_client = Mock()
    custom_client.converse.return_value = {"output": {"message": {"content": [{"text": "response"}]}}}

    with (
        patch("any_llm.providers.bedrock.bedrock._convert_response"),
        patch("boto3.client"),
    ):
        provider = BedrockProvider(client=custom_client, region_name="us-west-2")

        assert provider.client is custom_client
        assert provider.kwargs == {"region_name": "us-west-2"}


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


def test_completion_streaming_uses_converse_stream_on_resolved_client() -> None:
    """Test that streaming completions call converse_stream on the client resolved for the call.

    Guards against a regression where the timeout-aware client resolution (`_client_for_timeout`)
    is only wired into the non-streaming `converse` call and not `converse_stream`. Uses distinct
    base/timeout-scoped client mocks (rather than relying on `Mock().return_value` being the same
    object for every construction call) so the assertion would actually fail if `converse_stream`
    were called on the wrong client.
    """
    with mock_aws_provider() as mock_client_call:
        base_client = Mock()
        timeout_client = Mock()
        timeout_client.converse_stream.return_value = {"stream": [{"messageStart": {"role": "assistant"}}]}
        mock_client_call.side_effect = [base_client, timeout_client]

        provider = BedrockProvider(api_key="test_key")
        chunks = list(
            provider._completion(
                CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "Hello"}], stream=True),
                timeout=5,
            )
        )

        timeout_client.converse_stream.assert_called_once()
        base_client.converse_stream.assert_not_called()
        assert len(chunks) == 1


@pytest.mark.parametrize("reasoning_effort", [None, *get_args(ReasoningEffort)])
def test_completion_with_custom_reasoning_effort(reasoning_effort: ReasoningEffort | None) -> None:
    """Test that reasoning_effort is correctly passed to Bedrock API."""
    model_id = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_aws_provider() as mock_boto3_client:
        provider = BedrockProvider(api_key="test_key")
        provider._completion(
            CompletionParams(
                model_id=model_id,
                messages=messages,
                reasoning_effort=reasoning_effort,
            ),
        )

        call_kwargs = mock_boto3_client.return_value.converse.call_args[1]

        if reasoning_effort is None or reasoning_effort in ("none", "auto"):
            assert "additionalModelRequestFields" not in call_kwargs
        else:
            assert "additionalModelRequestFields" in call_kwargs
            assert call_kwargs["additionalModelRequestFields"]["reasoning_config"]["type"] == "enabled"
            assert (
                call_kwargs["additionalModelRequestFields"]["reasoning_config"]["budget_tokens"]
                == REASONING_EFFORT_TO_THINKING_BUDGETS[reasoning_effort]
            )


@contextmanager
def mock_aws_embedding_provider():  # type: ignore[no-untyped-def]
    """Mock AWS provider specifically for embedding tests."""
    with patch("boto3.Session") as mock_session_cls:
        mock_client = Mock()
        mock_session_cls.return_value.client.return_value = mock_client
        yield mock_session_cls.return_value.client, mock_client


def test_embedding_single_string() -> None:
    """Test embedding with a single string input."""
    model_id = "amazon.titan-embed-text-v1"
    input_text = "Hello world"

    mock_response_body = {"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 5}

    with mock_aws_embedding_provider() as (mock_client_call, mock_client):
        mock_client.invoke_model.return_value = {"body": Mock(read=Mock(return_value=json.dumps(mock_response_body)))}

        provider = BedrockProvider(api_key="test_key")
        response = provider._embedding(model_id, input_text)

        mock_client_call.assert_called_once()
        call_args, call_kwargs = mock_client_call.call_args
        assert call_args == ("bedrock-runtime",)
        assert call_kwargs["endpoint_url"] is None
        assert call_kwargs["config"].signature_version == "bearer"

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

    with mock_aws_embedding_provider() as (mock_client_call, mock_client):
        mock_client.invoke_model.side_effect = [
            {"body": Mock(read=Mock(return_value=json.dumps(mock_response_bodies[0])))},
            {"body": Mock(read=Mock(return_value=json.dumps(mock_response_bodies[1])))},
        ]

        provider = BedrockProvider(api_key="test_key")
        response = provider._embedding(model_id, input_texts)

        mock_client_call.assert_called_once()
        call_args, call_kwargs = mock_client_call.call_args
        assert call_args == ("bedrock-runtime",)
        assert call_kwargs["endpoint_url"] is None
        assert call_kwargs["config"].signature_version == "bearer"

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


def test_convert_images_for_bedrock_with_base64_image() -> None:
    """Test converting base64 image from OpenAI format to Bedrock format."""
    test_image_data = b"test image bytes"
    base64_data = base64.b64encode(test_image_data).decode("utf-8")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}},
    ]

    result = _convert_images_for_bedrock(content)

    assert len(result) == 2
    assert result[0] == {"text": "What is in this image?"}
    assert result[1]["image"]["format"] == "png"
    assert result[1]["image"]["source"]["bytes"] == test_image_data


def test_convert_images_for_bedrock_with_jpeg_image() -> None:
    """Test converting JPEG image from OpenAI format to Bedrock format."""
    test_image_data = b"jpeg image bytes"
    base64_data = base64.b64encode(test_image_data).decode("utf-8")

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}},
    ]

    result = _convert_images_for_bedrock(content)

    assert len(result) == 1
    assert result[0]["image"]["format"] == "jpeg"
    assert result[0]["image"]["source"]["bytes"] == test_image_data


def test_convert_images_for_bedrock_with_multiple_images() -> None:
    """Test converting multiple images in a single message."""
    image1_data = b"image one"
    image2_data = b"image two"
    base64_data1 = base64.b64encode(image1_data).decode("utf-8")
    base64_data2 = base64.b64encode(image2_data).decode("utf-8")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Compare these images."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data1}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{base64_data2}"}},
    ]

    result = _convert_images_for_bedrock(content)

    assert len(result) == 3
    assert result[0] == {"text": "Compare these images."}
    assert result[1]["image"]["format"] == "png"
    assert result[1]["image"]["source"]["bytes"] == image1_data
    assert result[2]["image"]["format"] == "webp"
    assert result[2]["image"]["source"]["bytes"] == image2_data


def test_convert_images_for_bedrock_raises_for_url_images() -> None:
    """Test that URL-based images raise InvalidRequestError."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]

    with pytest.raises(InvalidRequestError, match="URL-based images are not supported"):
        _convert_images_for_bedrock(content)


def test_convert_images_for_bedrock_text_only() -> None:
    """Test converting content with only text blocks."""
    content = [
        {"type": "text", "text": "Hello world"},
    ]

    result = _convert_images_for_bedrock(content)

    assert len(result) == 1
    assert result[0] == {"text": "Hello world"}


def test_convert_images_for_bedrock_raises_for_malformed_data_uri_missing_semicolon() -> None:
    """Test that malformed data URI without semicolon raises InvalidRequestError."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": "data:image/pngbase64,abc123"}},
    ]

    with pytest.raises(InvalidRequestError, match="missing semicolon separator"):
        _convert_images_for_bedrock(content)


def test_convert_images_for_bedrock_raises_for_missing_base64_marker() -> None:
    """Test that data URI without base64 marker raises InvalidRequestError."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": "data:image/png;abc123"}},
    ]

    with pytest.raises(InvalidRequestError, match="missing 'base64,' marker"):
        _convert_images_for_bedrock(content)


def test_convert_images_for_bedrock_raises_for_unsupported_format() -> None:
    """Test that unsupported image formats raise InvalidRequestError."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": "data:image/bmp;base64,abc123"}},
    ]

    with pytest.raises(InvalidRequestError, match="Unsupported image format: 'bmp'"):
        _convert_images_for_bedrock(content)


def test_convert_images_for_bedrock_raises_for_invalid_base64() -> None:
    """Test that invalid base64 data raises InvalidRequestError."""
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-valid-base64!!!"}},
    ]

    with pytest.raises(InvalidRequestError, match="Invalid base64 image data"):
        _convert_images_for_bedrock(content)


def test_convert_images_for_bedrock_normalizes_jpg_to_jpeg() -> None:
    """Test that jpg format is normalized to jpeg for Bedrock compatibility."""
    test_image_data = b"jpg image bytes"
    base64_data = base64.b64encode(test_image_data).decode("utf-8")

    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_data}"}},
    ]

    result = _convert_images_for_bedrock(content)

    assert len(result) == 1
    assert result[0]["image"]["format"] == "jpeg"
    assert result[0]["image"]["source"]["bytes"] == test_image_data


def test_convert_messages_with_image_content() -> None:
    """Test that _convert_messages correctly handles messages with image content."""
    test_image_data = b"test image"
    base64_data = base64.b64encode(test_image_data).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}},
            ],
        }
    ]

    system_message, formatted_messages = _convert_messages(messages)

    assert system_message == []
    assert len(formatted_messages) == 1
    assert formatted_messages[0]["role"] == "user"
    assert len(formatted_messages[0]["content"]) == 2
    assert formatted_messages[0]["content"][0] == {"text": "What is in this image?"}
    assert formatted_messages[0]["content"][1]["image"]["format"] == "png"
    assert formatted_messages[0]["content"][1]["image"]["source"]["bytes"] == test_image_data


def test_convert_messages_with_string_content() -> None:
    """Test that _convert_messages still works with simple string content."""
    messages = [{"role": "user", "content": "Hello world"}]

    system_message, formatted_messages = _convert_messages(messages)

    assert system_message == []
    assert len(formatted_messages) == 1
    assert formatted_messages[0] == {"role": "user", "content": [{"text": "Hello world"}]}


def test_completion_with_images() -> None:
    """Test that completion correctly processes image content."""
    test_image_data = b"test image bytes"
    base64_data = base64.b64encode(test_image_data).decode("utf-8")

    model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_data}"}},
            ],
        }
    ]

    with mock_aws_provider() as mock_boto3_client:
        provider = BedrockProvider(api_key="test_key")
        provider._completion(CompletionParams(model_id=model_id, messages=messages))

        call_args = mock_boto3_client.return_value.converse.call_args[1]
        assert call_args["modelId"] == model_id
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert len(call_args["messages"][0]["content"]) == 2
        assert call_args["messages"][0]["content"][0] == {"text": "Describe this image."}
        assert call_args["messages"][0]["content"][1]["image"]["format"] == "png"
        assert call_args["messages"][0]["content"][1]["image"]["source"]["bytes"] == test_image_data


def test_convert_response_extracts_cached_tokens() -> None:
    """Test that cacheReadInputTokens is extracted into prompt_tokens_details."""
    response: dict[str, Any] = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 50,
            "totalTokens": 150,
            "cacheReadInputTokens": 80,
            "cacheWriteInputTokens": 20,
        },
    }

    result = _convert_response(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 200  # 100 + 80 + 20
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 250  # 200 + 50
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 80


def test_convert_response_without_cached_tokens() -> None:
    """Test that prompt_tokens_details is None when no cached tokens are present."""
    response: dict[str, Any] = {
        "output": {"message": {"content": [{"text": "Hello!"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 50,
            "totalTokens": 150,
        },
    }

    result = _convert_response(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.prompt_tokens_details is None


def test_convert_response_tool_calls_extracts_cached_tokens() -> None:
    """Test that cached tokens are extracted for tool call responses."""
    response: dict[str, Any] = {
        "output": {
            "message": {
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tool-123",
                            "name": "get_weather",
                            "input": {"location": "Paris"},
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 50,
            "totalTokens": 150,
            "cacheReadInputTokens": 80,
        },
    }

    result = _convert_response(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 180  # 100 + 80
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 80


def test_streaming_metadata_chunk_extracts_cached_tokens() -> None:
    """Test that the metadata streaming event extracts cached tokens into usage."""
    chunk: dict[str, Any] = {
        "metadata": {
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
                "cacheReadInputTokens": 80,
                "cacheWriteInputTokens": 20,
            }
        }
    }

    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 200  # 100 + 80 + 20
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 250
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 80


def test_streaming_metadata_chunk_without_cached_tokens() -> None:
    """Test that streaming metadata works without cached tokens."""
    chunk: dict[str, Any] = {
        "metadata": {
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
            }
        }
    }

    result = _create_openai_chunk_from_aws_chunk(chunk, "test-model")

    assert result is not None
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.prompt_tokens_details is None


def test_convert_tool_spec_none_parameters() -> None:
    """Regression: parameters=None must not pass None to Bedrock's inputSchema."""
    tool_config = _convert_tool_spec(
        [{"type": "function", "function": {"name": "ping", "parameters": None}}],
        tool_choice=None,
    )
    spec = tool_config["tools"][0]["toolSpec"]
    assert spec["name"] == "ping"
    assert spec["inputSchema"]["json"] is not None
    assert spec["inputSchema"]["json"]["properties"] == {}


def test_convert_tool_spec_with_parameters() -> None:
    """Parameters present must be forwarded as-is to inputSchema."""
    params = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    tool_config = _convert_tool_spec(
        [{"type": "function", "function": {"name": "search", "parameters": params}}],
        tool_choice=None,
    )
    assert tool_config["tools"][0]["toolSpec"]["inputSchema"]["json"] == params


def test_convert_tool_spec_empty_description() -> None:
    """Regression: an empty description ("") must be coerced to a non-empty string.

    Bedrock's Converse API requires toolSpec.description to have a minimum length of 1.
    LangChain serializes tools without a docstring as ``"description": ""``; ``.get(default)``
    only fires when the key is missing, so the empty string must be coerced explicitly.
    """
    # (a) description present but empty -> coerced to " "
    tool_config = _convert_tool_spec(
        [{"type": "function", "function": {"name": "ping", "description": "", "parameters": None}}],
        tool_choice=None,
    )
    assert tool_config["tools"][0]["toolSpec"]["description"] == " "

    # (b) description key missing -> existing behaviour preserved
    tool_config = _convert_tool_spec(
        [{"type": "function", "function": {"name": "ping", "parameters": None}}],
        tool_choice=None,
    )
    assert tool_config["tools"][0]["toolSpec"]["description"] == " "
