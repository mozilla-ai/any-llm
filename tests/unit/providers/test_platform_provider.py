from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx
import pytest
from any_llm_platform_client import DecryptedProviderKey
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.openai import OpenaiProvider
from any_llm.providers.platform import PlatformProvider
from any_llm.providers.platform.utils import export_completion_trace
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionParams,
    CompletionUsage,
)
from any_llm.types.provider import PlatformKey


@pytest.fixture
def any_llm_key() -> str:
    """Fixture for a valid ANY_LLM_KEY."""
    return "ANY.v1.kid123.fingerprint456-base64key"


@pytest.fixture
def mock_decrypted_provider_key() -> DecryptedProviderKey:
    """Fixture for a mock DecryptedProviderKey."""
    return DecryptedProviderKey(
        api_key="mock-provider-api-key",
        provider_key_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        project_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        provider="openai",
        created_at=datetime.now(),
    )


@pytest.fixture
def mock_platform_provider(
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> PlatformProvider:
    """Fixture to create a mock platform provider with OpenAI (lazily initialized)."""
    provider = PlatformProvider(api_key=any_llm_key)
    provider.provider = OpenaiProvider
    return provider


async def _init_provider(
    provider: PlatformProvider,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Helper to trigger lazy async initialization with a mocked key."""
    with patch.object(
        provider.platform_client,
        "aget_decrypted_provider_key",
        new_callable=AsyncMock,
        return_value=mock_decrypted_provider_key,
    ):
        await provider._ensure_provider_initialized()


@pytest.fixture
def mock_completion() -> ChatCompletion:
    """Fixture for a mock ChatCompletion."""
    return ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4",
        created=1234567890,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Hello, world!"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


@pytest.fixture
def mock_platform_client() -> Mock:
    """Fixture for a mock AnyLLMPlatformClient."""
    from any_llm_platform_client import AnyLLMPlatformClient

    mock_client = Mock(spec=AnyLLMPlatformClient)
    mock_client._aensure_valid_token = AsyncMock(return_value="mock-jwt-token-12345")
    return mock_client


@pytest.fixture
def mock_streaming_chunks() -> list[ChatCompletionChunk]:
    """Fixture for mock streaming chunks with usage data."""
    return [
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="test-model",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        ),
    ]


def _get_single_span(exporter: InMemorySpanExporter):  # type: ignore[type-arg]
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def test_platform_key_valid_format() -> None:
    """Test that PlatformKey accepts valid API key formats."""
    valid_keys = [
        "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=",
        "ANY.v2.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY=",
    ]

    for key in valid_keys:
        platform_key = PlatformKey(api_key=key)
        assert platform_key.api_key == key


def test_platform_key_invalid_format_missing_prefix() -> None:
    """Test that PlatformKey rejects keys without the ANY prefix."""
    invalid_key = "NOT.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_invalid_format_missing_version() -> None:
    """Test that PlatformKey rejects keys without a version."""
    invalid_key = "ANY.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_invalid_format_missing_kid() -> None:
    """Test that PlatformKey rejects keys without a kid."""
    invalid_key = "ANY.v1.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_invalid_format_missing_fingerprint() -> None:
    """Test that PlatformKey rejects keys without a fingerprint."""
    invalid_key = "ANY.v1.kid123.-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_invalid_format_missing_base64_key() -> None:
    """Test that PlatformKey rejects keys without a base64 key."""
    invalid_key = "ANY.v1.kid123.fingerprint456-"

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_invalid_format_missing_separator() -> None:
    """Test that PlatformKey rejects keys without the dash separator."""
    invalid_key = "ANY.v1.kid123.fingerprint456YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_invalid_format_wrong_version_format() -> None:
    """Test that PlatformKey rejects keys with invalid version format."""
    invalid_key = "ANY.va.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key=invalid_key)

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_empty_string() -> None:
    """Test that PlatformKey rejects empty strings."""
    with pytest.raises(ValidationError) as exc_info:
        PlatformKey(api_key="")

    assert "Invalid API key format" in str(exc_info.value)


def test_platform_key_completely_invalid() -> None:
    """Test that PlatformKey rejects completely invalid strings."""
    invalid_keys = [
        "random-string",
        "123456",
        "sk-proj-1234567890",
    ]

    for invalid_key in invalid_keys:
        with pytest.raises(ValidationError) as exc_info:
            PlatformKey(api_key=invalid_key)

        assert "Invalid API key format" in str(exc_info.value)


def test_provider_setter_stores_class_without_http(
    any_llm_key: str,
) -> None:
    """Test that the provider setter stores the class lazily without making HTTP calls."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    assert provider_instance._provider_class is OpenaiProvider
    assert provider_instance._provider_initialized is False
    assert provider_instance._provider is None


@pytest.mark.asyncio
async def test_ensure_provider_initialized_uses_async_http(
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _ensure_provider_initialized uses async HTTP and creates the provider instance."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    with patch.object(
        provider_instance.platform_client,
        "aget_decrypted_provider_key",
        new_callable=AsyncMock,
        return_value=mock_decrypted_provider_key,
    ) as mock_aget_key:
        await provider_instance._ensure_provider_initialized()

        mock_aget_key.assert_called_once_with(any_llm_key=any_llm_key, provider="openai")

    assert provider_instance._provider_initialized is True
    assert provider_instance.provider.PROVIDER_NAME == "openai"
    assert provider_instance.provider_key_id == "550e8400-e29b-41d4-a716-446655440000"
    assert provider_instance.project_id == "550e8400-e29b-41d4-a716-446655440001"


@pytest.mark.asyncio
async def test_ensure_provider_initialized_is_idempotent(
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _ensure_provider_initialized only runs once."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    with patch.object(
        provider_instance.platform_client,
        "aget_decrypted_provider_key",
        new_callable=AsyncMock,
        return_value=mock_decrypted_provider_key,
    ) as mock_aget_key:
        await provider_instance._ensure_provider_initialized()
        await provider_instance._ensure_provider_initialized()

        mock_aget_key.assert_called_once()


def test_provider_getter_raises_before_init(
    any_llm_key: str,
) -> None:
    """Test that accessing .provider before async init raises RuntimeError."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    with pytest.raises(RuntimeError, match="Provider not yet initialized"):
        _ = provider_instance.provider


def test_prepare_creates_provider_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error handling when instantiating a PlatformProvider without an ANY_LLM_KEY set."""
    monkeypatch.delenv("ANY_LLM_KEY", raising=False)
    with pytest.raises(MissingApiKeyError):
        PlatformProvider()


def test_supports_flags_delegate_to_provider_class(
    any_llm_key: str,
) -> None:
    """Test that SUPPORTS_* flags reflect the wrapped provider class capabilities."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    assert provider_instance.SUPPORTS_COMPLETION is True
    assert provider_instance.SUPPORTS_COMPLETION_STREAMING is True
    assert provider_instance.SUPPORTS_LIST_MODELS is True
    assert provider_instance.SUPPORTS_EMBEDDING is True


def test_supports_flags_reflect_wrapped_provider_capabilities(
    any_llm_key: str,
) -> None:
    """Test that SUPPORTS_* flags accurately reflect providers with limited capabilities."""
    from any_llm.providers.anthropic.base import BaseAnthropicProvider

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = BaseAnthropicProvider

    assert provider_instance.SUPPORTS_EMBEDDING is False
    assert provider_instance.SUPPORTS_BATCH is False
    assert provider_instance.SUPPORTS_LIST_MODELS is False


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acompletion_non_streaming_success(
    mock_export_trace: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that non-streaming completions correctly call the provider and export traces."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    result = await provider_instance._acompletion(params, session_label="test-session")

    assert result == mock_completion
    provider_instance.provider._acompletion.assert_called_once_with(params=params)

    call_args = mock_export_trace.call_args
    assert call_args.kwargs["client"] == provider_instance.client
    assert call_args.kwargs["any_llm_key"] == any_llm_key
    assert call_args.kwargs["provider"] == "openai"
    assert call_args.kwargs["request_model"] == "gpt-4"
    assert call_args.kwargs["completion"] == mock_completion
    assert call_args.kwargs["session_label"] == "test-session"
    assert "platform_client" in call_args.kwargs


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acompletion_streaming_success(
    mock_export_trace: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that streaming completions correctly wrap the iterator and track usage."""
    mock_chunks = [
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=", world!"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        ),
    ]

    async def mock_stream():  # type: ignore[no-untyped-def]
        for chunk in mock_chunks:
            yield chunk

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    provider_instance.provider._acompletion = AsyncMock(return_value=mock_stream())  # type: ignore[method-assign, no-untyped-call]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    result = await provider_instance._acompletion(params)

    collected_chunks = []
    async for chunk in result:  # type: ignore[union-attr]
        collected_chunks.append(chunk)

    assert len(collected_chunks) == 3
    assert collected_chunks == mock_chunks
    provider_instance.provider._acompletion.assert_called_once_with(params=params)

    mock_export_trace.assert_called_once()
    call_args = mock_export_trace.call_args
    assert call_args.kwargs["client"] == provider_instance.client
    assert call_args.kwargs["any_llm_key"] == any_llm_key
    assert call_args.kwargs["provider"] == "openai"
    assert call_args.kwargs["request_model"] == "gpt-4"
    assert call_args.kwargs["completion"].usage.prompt_tokens == 10
    assert call_args.kwargs["completion"].usage.completion_tokens == 5
    assert call_args.kwargs["completion"].usage.total_tokens == 15
    assert "platform_client" in call_args.kwargs


@pytest.mark.asyncio
async def test_export_completion_trace_success(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test successful posting of completion trace."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    exporter = InMemorySpanExporter()
    test_provider = TracerProvider()
    test_provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = AsyncMock(spec=httpx.AsyncClient)

    with patch("any_llm.providers.platform.utils._get_or_create_tracer_provider", return_value=test_provider):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=any_llm_key,
            provider="openai",
            request_model="gpt-4",
            completion=mock_completion,
            start_time_ns=100,
            end_time_ns=200,
        )

    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)

    span = _get_single_span(exporter)
    assert span.kind.name == "CLIENT"
    assert span.start_time == 100
    assert span.end_time == 200
    assert span.attributes["gen_ai.provider.name"] == "openai"
    assert span.attributes["gen_ai.request.model"] == "gpt-4"
    assert span.attributes["gen_ai.response.model"] == "gpt-4"
    assert span.attributes["gen_ai.usage.input_tokens"] == 10
    assert span.attributes["gen_ai.usage.output_tokens"] == 5
    assert "anyllm.client_name" not in span.attributes


@pytest.mark.asyncio
async def test_export_completion_trace_with_client_name(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion trace with client_name included."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    client_name = "my-test-client"

    exporter = InMemorySpanExporter()
    test_provider = TracerProvider()
    test_provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = AsyncMock(spec=httpx.AsyncClient)

    with patch("any_llm.providers.platform.utils._get_or_create_tracer_provider", return_value=test_provider):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=any_llm_key,
            provider="openai",
            request_model="gpt-4",
            completion=mock_completion,
            start_time_ns=100,
            end_time_ns=200,
            client_name=client_name,
            session_label="session-1",
            conversation_id="user-123",
        )

    span = _get_single_span(exporter)
    assert span.attributes["anyllm.client_name"] == client_name
    assert span.attributes["anyllm.session_label"] == "session-1"
    assert span.attributes["gen_ai.conversation.id"] == "user-123"


@pytest.mark.asyncio
async def test_export_completion_trace_invalid_key_format() -> None:
    """Test error handling when ANY_LLM_KEY has invalid format."""
    from any_llm_platform_client import AnyLLMPlatformClient

    invalid_key = "INVALID_KEY_FORMAT"

    completion = ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4",
        created=1234567890,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )

    mock_platform_client = Mock(spec=AnyLLMPlatformClient)

    mock_platform_client._aensure_valid_token = AsyncMock(side_effect=ValueError("Invalid ANY_LLM_KEY format"))

    client = AsyncMock(spec=httpx.AsyncClient)

    with pytest.raises(ValueError, match="Invalid ANY_LLM_KEY format"):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=invalid_key,
            provider="openai",
            request_model="gpt-4",
            completion=completion,
            start_time_ns=100,
            end_time_ns=200,
        )


@patch("any_llm.any_llm.importlib.import_module")
def test_anyllm_instantiation_with_platform_key(
    mock_import_module: Mock,
) -> None:
    """Test that AnyLLM.create() correctly instantiates PlatformProvider when given a platform API key."""
    from any_llm import AnyLLM

    any_llm_key = "ANY.v1.kid123.fingerprint456-base64key"

    mock_provider_module = Mock()
    mock_provider_class = Mock()
    mock_provider_module.OpenaiProvider = mock_provider_class

    mock_platform_module = Mock()
    mock_platform_class = Mock(spec=PlatformProvider)
    mock_platform_instance = Mock(spec=PlatformProvider)
    mock_platform_class.return_value = mock_platform_instance
    mock_platform_module.PlatformProvider = mock_platform_class

    mock_import_module.side_effect = [mock_provider_module, mock_platform_module]

    result = AnyLLM.create(provider="openai", api_key=any_llm_key)

    assert result == mock_platform_instance
    assert mock_import_module.call_count == 2
    mock_import_module.assert_any_call("any_llm.providers.openai")
    mock_import_module.assert_any_call("any_llm.providers.platform")
    mock_platform_class.assert_called_once_with(api_key=any_llm_key, api_base=None, client_name=None)


@patch("any_llm.any_llm.importlib.import_module")
def test_anyllm_instantiation_with_platform_key_and_client_name(
    mock_import_module: Mock,
) -> None:
    """Test that AnyLLM.create() correctly passes client_name to PlatformProvider."""
    from any_llm import AnyLLM

    any_llm_key = "ANY.v1.kid123.fingerprint456-base64key"
    client_name = "test-client"

    mock_provider_module = Mock()
    mock_provider_class = Mock()
    mock_provider_module.OpenaiProvider = mock_provider_class

    mock_platform_module = Mock()
    mock_platform_class = Mock(spec=PlatformProvider)
    mock_platform_instance = Mock(spec=PlatformProvider)
    mock_platform_class.return_value = mock_platform_instance
    mock_platform_module.PlatformProvider = mock_platform_class

    mock_import_module.side_effect = [mock_provider_module, mock_platform_module]

    result = AnyLLM.create(provider="openai", api_key=any_llm_key, client_name=client_name)

    assert result == mock_platform_instance
    assert mock_import_module.call_count == 2
    mock_import_module.assert_any_call("any_llm.providers.openai")
    mock_import_module.assert_any_call("any_llm.providers.platform")
    mock_platform_class.assert_called_once_with(api_key=any_llm_key, api_base=None, client_name=client_name)


@patch("any_llm.any_llm.importlib.import_module")
def test_anyllm_instantiation_with_non_platform_key(
    mock_import_module: Mock,
) -> None:
    """Test that AnyLLM.create() falls through to regular provider when given a non-platform API key."""
    from any_llm import AnyLLM

    regular_api_key = "sk-proj-1234567890"

    mock_openai_module = Mock()
    mock_openai_class = Mock(spec=OpenaiProvider)
    mock_openai_instance = Mock(spec=OpenaiProvider)
    mock_openai_class.return_value = mock_openai_instance
    mock_openai_module.OpenaiProvider = mock_openai_class

    mock_import_module.return_value = mock_openai_module

    result = AnyLLM.create(provider="openai", api_key=regular_api_key)

    assert result == mock_openai_instance
    mock_import_module.assert_called_once_with("any_llm.providers.openai")
    mock_openai_class.assert_called_once_with(api_key=regular_api_key, api_base=None)


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_alist_models_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _alist_models delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_models = [Mock(), Mock()]
    provider_instance.provider._alist_models = AsyncMock(return_value=mock_models)  # type: ignore[method-assign]

    result = await provider_instance._alist_models()

    assert result == mock_models
    provider_instance.provider._alist_models.assert_called_once_with()


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_aembedding_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _aembedding delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_response = Mock()
    provider_instance.provider._aembedding = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

    result = await provider_instance._aembedding("text-embedding-ada-002", "hello")

    assert result == mock_response
    provider_instance.provider._aembedding.assert_called_once_with("text-embedding-ada-002", "hello")


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_aresponses_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _aresponses delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_response = Mock()
    provider_instance.provider._aresponses = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

    mock_params = Mock()
    result = await provider_instance._aresponses(mock_params)

    assert result == mock_response
    provider_instance.provider._aresponses.assert_called_once_with(mock_params)


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acreate_batch_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _acreate_batch delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_batch = Mock()
    provider_instance.provider._acreate_batch = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

    result = await provider_instance._acreate_batch(
        input_file_path="batch_input.jsonl",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"key": "value"},
    )

    assert result == mock_batch
    provider_instance.provider._acreate_batch.assert_called_once_with(
        input_file_path="batch_input.jsonl",
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"key": "value"},
    )


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_aretrieve_batch_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _aretrieve_batch delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_batch = Mock()
    provider_instance.provider._aretrieve_batch = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

    result = await provider_instance._aretrieve_batch("batch-123")

    assert result == mock_batch
    provider_instance.provider._aretrieve_batch.assert_called_once_with("batch-123")


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acancel_batch_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _acancel_batch delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_batch = Mock()
    provider_instance.provider._acancel_batch = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

    result = await provider_instance._acancel_batch("batch-123")

    assert result == mock_batch
    provider_instance.provider._acancel_batch.assert_called_once_with("batch-123")


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_alist_batches_delegates_to_provider(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that _alist_batches delegates to the wrapped provider."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    mock_batches = [Mock(), Mock()]
    provider_instance.provider._alist_batches = AsyncMock(return_value=mock_batches)  # type: ignore[method-assign]

    result = await provider_instance._alist_batches(after="batch-100", limit=10)

    assert result == mock_batches
    provider_instance.provider._alist_batches.assert_called_once_with(after="batch-100", limit=10)


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_delegation_triggers_lazy_init(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that calling a delegation method triggers lazy async initialization."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    assert provider_instance._provider_initialized is False

    with patch.object(
        provider_instance.platform_client,
        "aget_decrypted_provider_key",
        new_callable=AsyncMock,
        return_value=mock_decrypted_provider_key,
    ):
        mock_models = [Mock()]
        # We need to patch _alist_models on the OpenaiProvider instance that will be created
        with patch.object(OpenaiProvider, "_alist_models", new_callable=AsyncMock, return_value=mock_models):
            result = await provider_instance._alist_models()

    assert provider_instance._provider_initialized is True
    assert result == mock_models


@pytest.mark.asyncio
async def test_export_completion_trace_with_performance_metrics(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion trace with performance metrics included."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    exporter = InMemorySpanExporter()
    test_provider = TracerProvider()
    test_provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = AsyncMock(spec=httpx.AsyncClient)

    with patch("any_llm.providers.platform.utils._get_or_create_tracer_provider", return_value=test_provider):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=any_llm_key,
            provider="openai",
            request_model="gpt-4",
            completion=mock_completion,
            start_time_ns=100,
            end_time_ns=200,
            time_to_first_token_ms=50.0,
            time_to_last_token_ms=200.0,
            total_duration_ms=250.0,
            tokens_per_second=25.0,
            chunks_received=10,
            avg_chunk_size=0.5,
            inter_chunk_latency_variance_ms=5.0,
        )

    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)

    span = _get_single_span(exporter)
    assert span.attributes["anyllm.performance.time_to_first_token_ms"] == 50.0
    assert span.attributes["anyllm.performance.time_to_last_token_ms"] == 200.0
    assert span.attributes["anyllm.performance.total_duration_ms"] == 250.0
    assert span.attributes["anyllm.performance.tokens_per_second"] == 25.0
    assert span.attributes["anyllm.performance.chunks_received"] == 10
    assert span.attributes["anyllm.performance.avg_chunk_size"] == 0.5
    assert span.attributes["anyllm.performance.inter_chunk_latency_variance_ms"] == 5.0


@pytest.mark.asyncio
async def test_export_completion_trace_with_partial_performance_metrics(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion trace with only some performance metrics."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    exporter = InMemorySpanExporter()
    test_provider = TracerProvider()
    test_provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = AsyncMock(spec=httpx.AsyncClient)

    with patch("any_llm.providers.platform.utils._get_or_create_tracer_provider", return_value=test_provider):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=any_llm_key,
            provider="openai",
            request_model="gpt-4",
            completion=mock_completion,
            start_time_ns=100,
            end_time_ns=200,
            total_duration_ms=250.0,
            tokens_per_second=25.0,
        )

    span = _get_single_span(exporter)
    assert span.attributes["anyllm.performance.total_duration_ms"] == 250.0
    assert span.attributes["anyllm.performance.tokens_per_second"] == 25.0
    assert "anyllm.performance.time_to_first_token_ms" not in span.attributes
    assert "anyllm.performance.time_to_last_token_ms" not in span.attributes
    assert "anyllm.performance.chunks_received" not in span.attributes


@pytest.mark.asyncio
async def test_export_completion_trace_without_performance_metrics(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion trace without any performance metrics."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    exporter = InMemorySpanExporter()
    test_provider = TracerProvider()
    test_provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = AsyncMock(spec=httpx.AsyncClient)

    with patch("any_llm.providers.platform.utils._get_or_create_tracer_provider", return_value=test_provider):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=any_llm_key,
            provider="openai",
            request_model="gpt-4",
            completion=mock_completion,
            start_time_ns=100,
            end_time_ns=200,
        )

    span = _get_single_span(exporter)
    assert not any(key.startswith("anyllm.performance.") for key in span.attributes.keys())


@pytest.mark.asyncio
async def test_export_completion_trace_skips_when_no_usage(
    mock_platform_client: Mock,
) -> None:
    """Test that export_completion_trace still sends traces without usage attributes."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    completion = ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4",
        created=1234567890,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="Hello"),
                finish_reason="stop",
            )
        ],
        usage=None,
    )

    exporter = InMemorySpanExporter()
    test_provider = TracerProvider()
    test_provider.add_span_processor(SimpleSpanProcessor(exporter))

    client = AsyncMock(spec=httpx.AsyncClient)

    with patch("any_llm.providers.platform.utils._get_or_create_tracer_provider", return_value=test_provider):
        await export_completion_trace(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=any_llm_key,
            provider="openai",
            request_model="gpt-4",
            completion=completion,
            start_time_ns=100,
            end_time_ns=200,
        )

    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)
    span = _get_single_span(exporter)
    assert "gen_ai.usage.input_tokens" not in span.attributes
    assert "gen_ai.usage.output_tokens" not in span.attributes


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_streaming_performance_metrics_tracking(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that streaming completions correctly track performance metrics."""
    mock_chunks = [
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content=" world"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(content="!"),
                    finish_reason=None,
                )
            ],
        ),
        ChatCompletionChunk(
            id="chatcmpl-123",
            model="gpt-4",
            created=1234567890,
            object="chat.completion.chunk",
            choices=[
                ChunkChoice(
                    index=0,
                    delta=ChoiceDelta(),
                    finish_reason="stop",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        ),
    ]

    async def mock_stream():  # type: ignore[no-untyped-def]
        for chunk in mock_chunks:
            yield chunk

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)

    provider_instance.provider._acompletion = AsyncMock(return_value=mock_stream())  # type: ignore[method-assign, no-untyped-call]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    result = await provider_instance._acompletion(params)

    collected_chunks = []
    async for chunk in result:  # type: ignore[union-attr]
        collected_chunks.append(chunk)

    assert len(collected_chunks) == 4
    mock_post_usage.assert_called_once()

    call_args = mock_post_usage.call_args
    assert call_args.kwargs["time_to_first_token_ms"] is not None
    assert call_args.kwargs["time_to_last_token_ms"] is not None
    assert call_args.kwargs["total_duration_ms"] is not None
    assert call_args.kwargs["tokens_per_second"] is not None
    assert call_args.kwargs["chunks_received"] == 4
    assert call_args.kwargs["avg_chunk_size"] is not None
    # Inter-chunk latency variance requires at least 2 chunks
    assert call_args.kwargs["inter_chunk_latency_variance_ms"] is not None


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_non_streaming_includes_total_duration(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that non-streaming completions include total_duration_ms metric."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    await provider_instance._acompletion(params)

    mock_post_usage.assert_called_once()
    call_args = mock_post_usage.call_args
    assert call_args.kwargs["total_duration_ms"] is not None
    assert call_args.kwargs["total_duration_ms"] > 0


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_stream_options_auto_injected_when_not_set(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_streaming_chunks: list[ChatCompletionChunk],
) -> None:
    """Test that PlatformProvider auto-injects stream_options when not set by user."""

    async def mock_stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in mock_streaming_chunks:
            yield chunk

    provider_instance = PlatformProvider(api_key=any_llm_key)

    mock_provider = Mock()
    mock_provider.PROVIDER_NAME = LLMProvider.OPENAI
    provider_instance._provider = mock_provider
    provider_instance._provider_initialized = True

    captured_params = None

    async def capture_and_return(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal captured_params
        captured_params = kwargs.get("params") or args[0]
        return mock_stream()

    provider_instance.provider._acompletion = AsyncMock(side_effect=capture_and_return)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options=None,
    )

    result = await provider_instance._acompletion(params)

    async for _ in result:  # type: ignore[union-attr]
        pass

    assert captured_params is not None
    assert captured_params.stream_options == {"include_usage": True}

    # Verify original params not mutated
    assert params.stream_options is None


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_stream_options_preserved_when_user_specifies_it(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_streaming_chunks: list[ChatCompletionChunk],
) -> None:
    """Test that user-specified stream_options are preserved."""

    async def mock_stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in mock_streaming_chunks:
            yield chunk

    provider_instance = PlatformProvider(api_key=any_llm_key)

    mock_provider = Mock()
    mock_provider.PROVIDER_NAME = LLMProvider.OPENAI
    provider_instance._provider = mock_provider
    provider_instance._provider_initialized = True

    captured_params = None

    async def capture_and_return(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal captured_params
        captured_params = kwargs.get("params") or args[0]
        return mock_stream()

    provider_instance.provider._acompletion = AsyncMock(side_effect=capture_and_return)  # type: ignore[method-assign]

    custom_stream_options = {"include_usage": False, "custom_field": "custom_value"}
    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options=custom_stream_options,
    )

    result = await provider_instance._acompletion(params)

    async for _ in result:  # type: ignore[union-attr]
        pass

    assert captured_params is not None
    assert captured_params.stream_options == custom_stream_options


@pytest.mark.asyncio
async def test_trace_export_uses_bearer_token(
    mock_platform_client: Mock,
    any_llm_key: str,
    mock_completion: ChatCompletion,
) -> None:
    """Test that trace export uses Bearer token authentication."""
    from any_llm.providers.platform import utils as platform_utils

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    captured: dict[str, Any] = {}

    def _exporter_factory(*args: Any, **kwargs: Any) -> InMemorySpanExporter:  # type: ignore[type-arg]
        captured.update(kwargs)
        return InMemorySpanExporter()

    # Clear the provider cache so a new provider is created with our mocked exporter
    original_providers = platform_utils._providers.copy()
    platform_utils._providers.clear()
    try:
        with patch("any_llm.providers.platform.utils.OTLPSpanExporter", side_effect=_exporter_factory):
            await export_completion_trace(
                platform_client=mock_platform_client,
                client=mock_http_client,
                any_llm_key=any_llm_key,
                provider="openai",
                request_model="gpt-4",
                completion=mock_completion,
                start_time_ns=100,
                end_time_ns=200,
                client_name="test-client",
                total_duration_ms=100.0,
            )
    finally:
        # Shutdown any providers created during the test and restore original state
        for provider in platform_utils._providers.values():
            provider.shutdown()
        platform_utils._providers.clear()
        platform_utils._providers.update(original_providers)

    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)

    headers = captured["headers"]

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer mock-jwt-token-12345"
    assert "encryption-key" not in headers
    assert "AnyLLM-Challenge-Response" not in headers


@pytest.mark.asyncio
async def test_trace_export_includes_version_header(
    mock_platform_client: Mock,
    any_llm_key: str,
    mock_completion: ChatCompletion,
) -> None:
    """Test that trace export includes library version in User-Agent header."""
    from any_llm import __version__
    from any_llm.providers.platform import utils as platform_utils

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    captured: dict[str, Any] = {}

    def _exporter_factory(*args: Any, **kwargs: Any) -> InMemorySpanExporter:  # type: ignore[type-arg]
        captured.update(kwargs)
        return InMemorySpanExporter()

    # Clear the provider cache so a new provider is created with our mocked exporter
    original_providers = platform_utils._providers.copy()
    platform_utils._providers.clear()
    try:
        with patch("any_llm.providers.platform.utils.OTLPSpanExporter", side_effect=_exporter_factory):
            await export_completion_trace(
                platform_client=mock_platform_client,
                client=mock_http_client,
                any_llm_key=any_llm_key,
                provider="openai",
                request_model="gpt-4",
                completion=mock_completion,
                start_time_ns=100,
                end_time_ns=200,
            )
    finally:
        for provider in platform_utils._providers.values():
            provider.shutdown()
        platform_utils._providers.clear()
        platform_utils._providers.update(original_providers)

    headers = captured["headers"]

    assert "User-Agent" in headers
    assert headers["User-Agent"] == f"python-any-llm/{__version__}"


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acompletion_rejects_client_name_in_kwargs_when_not_initialized(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that _acompletion rejects request-time client_name when provider was not initialized with it."""
    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    client_name = "test-client-from-kwargs"

    with pytest.raises(ValueError, match="Passing client_name at request time is not supported"):
        await provider_instance._acompletion(params, client_name=client_name)

    provider_instance.provider._acompletion.assert_not_called()
    mock_post_usage.assert_not_called()


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acompletion_rejects_changing_existing_client_name(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that _acompletion rejects a request-time client_name that differs from configured client_name."""
    initial_client_name = "initial-client"
    provider_instance = PlatformProvider(api_key=any_llm_key, client_name=initial_client_name)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    new_client_name = "new-client-from-kwargs"

    with pytest.raises(ValueError, match="Passing client_name at request time is not supported"):
        await provider_instance._acompletion(params, client_name=new_client_name)

    provider_instance.provider._acompletion.assert_not_called()
    mock_post_usage.assert_not_called()


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_acompletion_rejects_same_client_name_in_kwargs(
    mock_post_usage: AsyncMock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that _acompletion rejects request-time client_name even when it matches configured value."""
    client_name = "configured-client"
    provider_instance = PlatformProvider(api_key=any_llm_key, client_name=client_name)
    provider_instance.provider = OpenaiProvider
    await _init_provider(provider_instance, mock_decrypted_provider_key)
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    with pytest.raises(ValueError, match="Passing client_name at request time is not supported"):
        await provider_instance._acompletion(params, client_name=client_name)

    provider_instance.provider._acompletion.assert_not_called()
    mock_post_usage.assert_not_called()


# Tests for mzai provider with JWT token authentication


@pytest.mark.asyncio
async def test_platform_provider_with_mzai_fetches_token(
    any_llm_key: str,
) -> None:
    """Test that mzai provider fetches JWT token instead of decrypted provider key."""
    from any_llm.providers.mzai import MzaiProvider

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = MzaiProvider

    with patch.object(
        provider_instance.platform_client,
        "_aensure_valid_token",
        new_callable=AsyncMock,
        return_value="mock-jwt-token-12345",
    ) as mock_ensure_token:
        await provider_instance._ensure_provider_initialized()

    assert provider_instance.PROVIDER_NAME == "platform"
    assert provider_instance.provider.PROVIDER_NAME == "mzai"
    assert provider_instance.provider_key_id is None
    assert provider_instance.project_id is None

    mock_ensure_token.assert_called_once_with(any_llm_key)


@pytest.mark.asyncio
async def test_platform_provider_mzai_passes_token_to_provider(
    any_llm_key: str,
) -> None:
    """Test that the JWT token is passed as api_key to MzaiProvider."""
    from any_llm.providers.mzai import MzaiProvider

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = MzaiProvider

    with patch.object(
        provider_instance.platform_client,
        "_aensure_valid_token",
        new_callable=AsyncMock,
        return_value="mock-jwt-token-12345",
    ):
        await provider_instance._ensure_provider_initialized()

    # The underlying provider should have the JWT token as its API key
    assert isinstance(provider_instance.provider, MzaiProvider)
    assert provider_instance.provider.client.api_key == "mock-jwt-token-12345"


async def _init_mzai_provider(provider: PlatformProvider) -> None:
    """Helper to trigger lazy async initialization for mzai with a mocked JWT token."""
    with patch.object(
        provider.platform_client,
        "_aensure_valid_token",
        new_callable=AsyncMock,
        return_value="mock-jwt-token-12345",
    ):
        await provider._ensure_provider_initialized()


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_platform_provider_mzai_exports_traces_non_streaming(
    mock_export_trace: AsyncMock,
    any_llm_key: str,
    mock_completion: ChatCompletion,
) -> None:
    """Test that traces are exported for mzai provider (non-streaming)."""
    from any_llm.providers.mzai import MzaiProvider

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = MzaiProvider
    await _init_mzai_provider(provider_instance)
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    result = await provider_instance._acompletion(params)

    assert result == mock_completion
    mock_export_trace.assert_called_once()
    assert mock_export_trace.call_args.kwargs["provider"] == "mzai"


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.export_completion_trace")
async def test_platform_provider_mzai_exports_traces_streaming(
    mock_export_trace: AsyncMock,
    any_llm_key: str,
    mock_streaming_chunks: list[ChatCompletionChunk],
) -> None:
    """Test that traces are exported for mzai provider (streaming)."""
    from any_llm.providers.mzai import MzaiProvider

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = MzaiProvider
    await _init_mzai_provider(provider_instance)

    async def mock_stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in mock_streaming_chunks:
            yield chunk

    provider_instance.provider._acompletion = AsyncMock(return_value=mock_stream())  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )

    result = await provider_instance._acompletion(params)

    # Consume the stream
    chunks = []
    async for chunk in result:  # type: ignore[union-attr]
        chunks.append(chunk)

    assert len(chunks) == len(mock_streaming_chunks)
    mock_export_trace.assert_called_once()
    assert mock_export_trace.call_args.kwargs["provider"] == "mzai"


def test_tracer_provider_reused_for_same_token() -> None:
    """Test that _get_or_create_tracer_provider returns the same provider for the same token."""
    from any_llm.providers.platform import utils as platform_utils
    from any_llm.providers.platform.utils import _get_or_create_tracer_provider

    original_providers = platform_utils._providers.copy()
    platform_utils._providers.clear()
    try:
        provider_a = _get_or_create_tracer_provider("token-aaa")
        provider_b = _get_or_create_tracer_provider("token-aaa")
        assert provider_a is provider_b

        provider_c = _get_or_create_tracer_provider("token-bbb")
        assert provider_c is not provider_a
    finally:
        for provider in platform_utils._providers.values():
            provider.shutdown()
        platform_utils._providers.clear()
        platform_utils._providers.update(original_providers)


def test_shutdown_telemetry_clears_providers() -> None:
    """Test that shutdown_telemetry shuts down all providers and clears the cache."""
    from any_llm.providers.platform import utils as platform_utils
    from any_llm.providers.platform.utils import _get_or_create_tracer_provider, shutdown_telemetry

    original_providers = platform_utils._providers.copy()
    platform_utils._providers.clear()
    try:
        _get_or_create_tracer_provider("token-xxx")
        _get_or_create_tracer_provider("token-yyy")
        assert len(platform_utils._providers) == 2

        shutdown_telemetry()

        assert len(platform_utils._providers) == 0
    finally:
        # Restore original state (shutdown_telemetry already cleared, but be safe)
        platform_utils._providers.clear()
        platform_utils._providers.update(original_providers)
