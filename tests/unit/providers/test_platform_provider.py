from collections.abc import AsyncIterator
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

import httpx
import pytest
from any_llm_platform_client import DecryptedProviderKey
from pydantic import ValidationError

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.openai import OpenaiProvider
from any_llm.providers.platform import PlatformProvider
from any_llm.providers.platform.utils import post_completion_usage_event
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


# Fixtures
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
    """Fixture to create a mock platform provider with OpenAI."""
    with patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key") as mock_get_key:
        mock_get_key.return_value = mock_decrypted_provider_key
        provider = PlatformProvider(api_key=any_llm_key)
        provider.provider = OpenaiProvider
        return provider


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


@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
def test_prepare_creates_provider(
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test proper initialization with an API key."""
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider

    assert provider_instance.PROVIDER_NAME == "platform"
    assert provider_instance.provider.PROVIDER_NAME == "openai"
    assert provider_instance.provider_key_id == "550e8400-e29b-41d4-a716-446655440000"
    assert provider_instance.project_id == "550e8400-e29b-41d4-a716-446655440001"

    # Verify get_decrypted_provider_key was called
    call_args = mock_get_decrypted_provider_key.call_args
    assert call_args.kwargs["any_llm_key"] == any_llm_key
    assert call_args.kwargs["provider"] == "openai"


def test_prepare_creates_provider_without_api_key() -> None:
    """Test error handling when instantiating a PlatformProvider without an ANY_LLM_KEY set."""
    with pytest.raises(MissingApiKeyError):
        PlatformProvider()


@pytest.mark.asyncio
@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_acompletion_non_streaming_success(
    mock_post_usage: AsyncMock,
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that non-streaming completions correctly call the provider and post usage events."""
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    provider_instance = PlatformProvider(api_key=any_llm_key)
    provider_instance.provider = OpenaiProvider
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    # Create completion params
    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    # Call _acompletion
    result = await provider_instance._acompletion(params)

    # Assertions
    assert result == mock_completion
    provider_instance.provider._acompletion.assert_called_once_with(params=params)

    # Verify post_completion_usage_event was called with the platform_client instance
    call_args = mock_post_usage.call_args
    assert call_args.kwargs["client"] == provider_instance.client
    assert call_args.kwargs["any_llm_key"] == any_llm_key
    assert call_args.kwargs["provider"] == "openai"
    assert call_args.kwargs["completion"] == mock_completion
    assert call_args.kwargs["provider_key_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert "platform_client" in call_args.kwargs


@pytest.mark.asyncio
@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_acompletion_streaming_success(
    mock_post_usage: AsyncMock,
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that streaming completions correctly wrap the iterator and track usage."""
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    # Create mock streaming chunks
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

    provider_instance = PlatformProvider(
        api_key=any_llm_key,
    )
    provider_instance.provider = OpenaiProvider

    # Mock the underlying provider's _acompletion method
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_stream())  # type: ignore[method-assign, no-untyped-call]

    # Create completion params
    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    # Call _acompletion
    result = await provider_instance._acompletion(params)

    # Collect all chunks from the stream
    collected_chunks = []
    async for chunk in result:  # type: ignore[union-attr]
        collected_chunks.append(chunk)

    # Assertions
    assert len(collected_chunks) == 3
    assert collected_chunks == mock_chunks
    provider_instance.provider._acompletion.assert_called_once_with(params=params)

    # Verify usage event was posted with correct data
    mock_post_usage.assert_called_once()
    call_args = mock_post_usage.call_args
    assert call_args.kwargs["client"] == provider_instance.client
    assert call_args.kwargs["any_llm_key"] == any_llm_key
    assert call_args.kwargs["provider"] == "openai"
    assert call_args.kwargs["completion"].usage.prompt_tokens == 10
    assert call_args.kwargs["completion"].usage.completion_tokens == 5
    assert call_args.kwargs["completion"].usage.total_tokens == 15
    assert call_args.kwargs["provider_key_id"] == "550e8400-e29b-41d4-a716-446655440000"


@pytest.mark.asyncio
async def test_post_completion_usage_event_success(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test successful posting of completion usage event."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    provider_key_id = UUID("550e8400-e29b-41d4-a716-446655440002")

    # Create mock httpx client
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)

    # Call the function
    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id=str(provider_key_id),
    )

    # Assertions
    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)

    # Usage event POST should be called once
    client.post.assert_called_once()

    # Verify the payload sent to the usage event endpoint
    call_args = client.post.call_args
    assert "/usage-events/" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["provider_key_id"] == str(provider_key_id)
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4"
    assert payload["data"]["input_tokens"] == "10"
    assert payload["data"]["output_tokens"] == "5"
    assert "id" in payload
    assert "client_name" not in payload  # No client_name provided


@pytest.mark.asyncio
async def test_post_completion_usage_event_with_client_name(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion usage event with client_name included."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    provider_key_id = UUID("550e8400-e29b-41d4-a716-446655440002")
    client_name = "my-test-client"

    # Create mock httpx client
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)

    # Call the function
    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id=str(provider_key_id),
        client_name=client_name,
    )

    # Verify client_name is included in the payload
    call_args = client.post.call_args
    payload = call_args.kwargs["json"]
    assert payload["client_name"] == client_name
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4"


@pytest.mark.asyncio
async def test_post_completion_usage_event_invalid_key_format() -> None:
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

    # Mock the token management method to raise ValueError for invalid key
    mock_platform_client._aensure_valid_token = AsyncMock(side_effect=ValueError("Invalid ANY_LLM_KEY format"))

    client = AsyncMock(spec=httpx.AsyncClient)

    with pytest.raises(ValueError, match="Invalid ANY_LLM_KEY format"):
        await post_completion_usage_event(
            platform_client=mock_platform_client,
            client=client,
            any_llm_key=invalid_key,
            provider="openai",
            completion=completion,
            provider_key_id="550e8400-e29b-41d4-a716-446655440000",
        )


@patch("any_llm.any_llm.importlib.import_module")
def test_anyllm_instantiation_with_platform_key(
    mock_import_module: Mock,
) -> None:
    """Test that AnyLLM.create() correctly instantiates PlatformProvider when given a platform API key."""
    from any_llm import AnyLLM

    any_llm_key = "ANY.v1.kid123.fingerprint456-base64key"

    # Mock the provider module first (for initial validation)
    mock_provider_module = Mock()
    mock_provider_class = Mock()
    mock_provider_module.OpenaiProvider = mock_provider_class

    # Mock the PlatformProvider module and class
    mock_platform_module = Mock()
    mock_platform_class = Mock(spec=PlatformProvider)
    mock_platform_instance = Mock(spec=PlatformProvider)
    mock_platform_class.return_value = mock_platform_instance
    mock_platform_module.PlatformProvider = mock_platform_class

    # Configure import_module to return provider module first, then platform module
    mock_import_module.side_effect = [mock_provider_module, mock_platform_module]

    # Call AnyLLM.create() with platform key
    result = AnyLLM.create(provider="openai", api_key=any_llm_key)

    # Assertions
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

    # Mock the provider module first (for initial validation)
    mock_provider_module = Mock()
    mock_provider_class = Mock()
    mock_provider_module.OpenaiProvider = mock_provider_class

    # Mock the PlatformProvider module and class
    mock_platform_module = Mock()
    mock_platform_class = Mock(spec=PlatformProvider)
    mock_platform_instance = Mock(spec=PlatformProvider)
    mock_platform_class.return_value = mock_platform_instance
    mock_platform_module.PlatformProvider = mock_platform_class

    # Configure import_module to return provider module first, then platform module
    mock_import_module.side_effect = [mock_provider_module, mock_platform_module]

    # Call AnyLLM.create() with platform key and client_name
    result = AnyLLM.create(provider="openai", api_key=any_llm_key, client_name=client_name)

    # Assertions
    assert result == mock_platform_instance
    assert mock_import_module.call_count == 2
    mock_import_module.assert_any_call("any_llm.providers.openai")
    mock_import_module.assert_any_call("any_llm.providers.platform")
    # Verify client_name is passed correctly and only once
    mock_platform_class.assert_called_once_with(api_key=any_llm_key, api_base=None, client_name=client_name)


@patch("any_llm.any_llm.importlib.import_module")
def test_anyllm_instantiation_with_non_platform_key(
    mock_import_module: Mock,
) -> None:
    """Test that AnyLLM.create() falls through to regular provider when given a non-platform API key."""
    from any_llm import AnyLLM

    regular_api_key = "sk-proj-1234567890"

    # Mock the OpenAI provider module and class
    mock_openai_module = Mock()
    mock_openai_class = Mock(spec=OpenaiProvider)
    mock_openai_instance = Mock(spec=OpenaiProvider)
    mock_openai_class.return_value = mock_openai_instance
    mock_openai_module.OpenaiProvider = mock_openai_class

    # Configure import_module to return our mock for OpenAI
    mock_import_module.return_value = mock_openai_module

    # Call AnyLLM.create() with regular key
    result = AnyLLM.create(provider="openai", api_key=regular_api_key)

    # Assertions
    assert result == mock_openai_instance
    mock_import_module.assert_called_once_with("any_llm.providers.openai")
    mock_openai_class.assert_called_once_with(api_key=regular_api_key, api_base=None)


@pytest.mark.asyncio
async def test_post_completion_usage_event_with_performance_metrics(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion usage event with performance metrics included."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    provider_key_id = UUID("550e8400-e29b-41d4-a716-446655440002")

    # Create mock httpx client
    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)

    # Call the function with performance metrics
    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id=str(provider_key_id),
        time_to_first_token_ms=50.0,
        time_to_last_token_ms=200.0,
        total_duration_ms=250.0,
        tokens_per_second=25.0,
        chunks_received=10,
        avg_chunk_size=0.5,
        inter_chunk_latency_variance_ms=5.0,
    )

    # Assertions
    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)
    client.post.assert_called_once()

    # Verify the payload includes performance metrics
    call_args = client.post.call_args
    payload = call_args.kwargs["json"]
    assert "performance" in payload["data"]
    performance = payload["data"]["performance"]
    assert performance["time_to_first_token_ms"] == 50.0
    assert performance["time_to_last_token_ms"] == 200.0
    assert performance["total_duration_ms"] == 250.0
    assert performance["tokens_per_second"] == 25.0
    assert performance["chunks_received"] == 10
    assert performance["avg_chunk_size"] == 0.5
    assert performance["inter_chunk_latency_variance_ms"] == 5.0


@pytest.mark.asyncio
async def test_post_completion_usage_event_with_partial_performance_metrics(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion usage event with only some performance metrics."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    provider_key_id = UUID("550e8400-e29b-41d4-a716-446655440002")

    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)

    # Call with only some performance metrics
    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id=str(provider_key_id),
        total_duration_ms=250.0,
        tokens_per_second=25.0,
    )

    # Verify only provided metrics are included
    call_args = client.post.call_args
    payload = call_args.kwargs["json"]
    assert "performance" in payload["data"]
    performance = payload["data"]["performance"]
    assert performance["total_duration_ms"] == 250.0
    assert performance["tokens_per_second"] == 25.0
    assert "time_to_first_token_ms" not in performance
    assert "time_to_last_token_ms" not in performance
    assert "chunks_received" not in performance


@pytest.mark.asyncio
async def test_post_completion_usage_event_without_performance_metrics(
    mock_platform_client: Mock,
    mock_completion: ChatCompletion,
) -> None:
    """Test posting completion usage event without any performance metrics."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    provider_key_id = UUID("550e8400-e29b-41d4-a716-446655440002")

    mock_response = Mock()
    mock_response.raise_for_status = Mock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)

    # Call without any performance metrics
    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id=str(provider_key_id),
    )

    # Verify performance section is not included when no metrics provided
    call_args = client.post.call_args
    payload = call_args.kwargs["json"]
    assert "performance" not in payload["data"]


@pytest.mark.asyncio
async def test_post_completion_usage_event_skips_when_no_usage(
    mock_platform_client: Mock,
) -> None:
    """Test that post_completion_usage_event returns early when completion has no usage data."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    # Create completion without usage data
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

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock()

    # Call the function
    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=completion,
        provider_key_id="550e8400-e29b-41d4-a716-446655440000",
    )

    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)
    client.post.assert_not_called()


@pytest.mark.asyncio
@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_streaming_performance_metrics_tracking(
    mock_post_usage: AsyncMock,
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
) -> None:
    """Test that streaming completions correctly track performance metrics."""
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    # Create mock streaming chunks with content
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

    provider_instance = PlatformProvider(
        api_key=any_llm_key,
    )
    provider_instance.provider = OpenaiProvider

    provider_instance.provider._acompletion = AsyncMock(return_value=mock_stream())  # type: ignore[method-assign, no-untyped-call]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    # Call _acompletion
    result = await provider_instance._acompletion(params)

    # Collect all chunks from the stream
    collected_chunks = []
    async for chunk in result:  # type: ignore[union-attr]
        collected_chunks.append(chunk)

    # Assertions
    assert len(collected_chunks) == 4
    mock_post_usage.assert_called_once()

    # Verify performance metrics were tracked
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
@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_non_streaming_includes_total_duration(
    mock_post_usage: AsyncMock,
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_completion: ChatCompletion,
) -> None:
    """Test that non-streaming completions include total_duration_ms metric."""
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    provider_instance = PlatformProvider(
        api_key=any_llm_key,
    )
    provider_instance.provider = OpenaiProvider
    provider_instance.provider._acompletion = AsyncMock(return_value=mock_completion)  # type: ignore[method-assign]

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    # Call _acompletion
    await provider_instance._acompletion(params)

    # Verify total_duration_ms was tracked
    mock_post_usage.assert_called_once()
    call_args = mock_post_usage.call_args
    assert call_args.kwargs["total_duration_ms"] is not None
    assert call_args.kwargs["total_duration_ms"] > 0


@pytest.mark.asyncio
@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
@pytest.mark.parametrize(
    "provider_name",
    [
        # Supported providers
        LLMProvider.OPENAI,
        LLMProvider.HUGGINGFACE,
        LLMProvider.GROQ,
        LLMProvider.DEEPSEEK,
        # Unsupported providers
        LLMProvider.ANTHROPIC,
        LLMProvider.CEREBRAS,
        LLMProvider.COHERE,
        LLMProvider.GEMINI,
        LLMProvider.MISTRAL,
        LLMProvider.OLLAMA,
        LLMProvider.TOGETHER,
    ],
)
async def test_stream_options_handling_by_provider(
    mock_post_usage: AsyncMock,
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_streaming_chunks: list[ChatCompletionChunk],
    provider_name: LLMProvider,
) -> None:
    """Test stream_options handling for both supported and unsupported providers."""
    # Define unsupported providers (should match platform.py)
    providers_without_stream_options = {
        LLMProvider.ANTHROPIC,
        LLMProvider.CEREBRAS,
        LLMProvider.COHERE,
        LLMProvider.GEMINI,
        LLMProvider.MISTRAL,
        LLMProvider.OLLAMA,
        LLMProvider.TOGETHER,
    }

    mock_decrypted_provider_key.provider = provider_name.value
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    async def mock_stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in mock_streaming_chunks:
            yield chunk

    provider_instance = PlatformProvider(api_key=any_llm_key)

    # Create a mock provider with the PROVIDER_NAME attribute
    mock_provider = Mock()
    mock_provider.PROVIDER_NAME = provider_name
    provider_instance._provider = mock_provider

    # Mock the underlying provider's _acompletion to capture the params it receives
    captured_params = None

    async def capture_and_return(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal captured_params
        captured_params = kwargs.get("params") or args[0]
        return mock_stream()

    provider_instance.provider._acompletion = AsyncMock(side_effect=capture_and_return)  # type: ignore[method-assign]

    # Create completion params WITHOUT stream_options
    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options=None,
    )

    # Call _acompletion
    result = await provider_instance._acompletion(params)

    # Consume the stream
    async for _ in result:  # type: ignore[union-attr]
        pass

    # Verify behavior based on provider support
    assert captured_params is not None
    if provider_name in providers_without_stream_options:
        # Unsupported providers should receive None
        assert captured_params.stream_options is None
    else:
        # Supported providers should have it auto-enabled
        assert captured_params.stream_options == {"include_usage": True}

    # Verify original params not mutated
    assert params.stream_options is None


@pytest.mark.asyncio
@patch("any_llm_platform_client.AnyLLMPlatformClient.get_decrypted_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_stream_options_preserved_when_user_specifies_it(
    mock_post_usage: AsyncMock,
    mock_get_decrypted_provider_key: Mock,
    any_llm_key: str,
    mock_decrypted_provider_key: DecryptedProviderKey,
    mock_streaming_chunks: list[ChatCompletionChunk],
) -> None:
    """Test that user-specified stream_options are preserved for supported providers."""
    mock_decrypted_provider_key.provider = LLMProvider.OPENAI.value
    mock_get_decrypted_provider_key.return_value = mock_decrypted_provider_key

    async def mock_stream() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in mock_streaming_chunks:
            yield chunk

    provider_instance = PlatformProvider(api_key=any_llm_key)

    # Create a mock provider with the PROVIDER_NAME attribute
    mock_provider = Mock()
    mock_provider.PROVIDER_NAME = LLMProvider.OPENAI
    provider_instance._provider = mock_provider

    # Mock the underlying provider's _acompletion to capture the params it receives
    captured_params = None

    async def capture_and_return(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal captured_params
        captured_params = kwargs.get("params") or args[0]
        return mock_stream()

    provider_instance.provider._acompletion = AsyncMock(side_effect=capture_and_return)  # type: ignore[method-assign]

    # User specifies custom stream_options
    custom_stream_options = {"include_usage": False, "custom_field": "custom_value"}
    params = CompletionParams(
        model_id="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        stream_options=custom_stream_options,
    )

    # Call _acompletion
    result = await provider_instance._acompletion(params)

    # Consume the stream
    async for _ in result:  # type: ignore[union-attr]
        pass

    # Verify that user-specified stream_options are preserved
    assert captured_params is not None
    assert captured_params.stream_options == custom_stream_options


@pytest.mark.asyncio
async def test_usage_event_uses_bearer_token(
    mock_platform_client: Mock,
    any_llm_key: str,
    mock_completion: ChatCompletion,
) -> None:
    """Test that usage events use Bearer token authentication (v3.0)."""
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_http_client.post = AsyncMock(return_value=mock_response)

    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=mock_http_client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id="550e8400-e29b-41d4-a716-446655440000",
        client_name="test-client",
        total_duration_ms=100.0,
    )

    mock_platform_client._aensure_valid_token.assert_called_once_with(any_llm_key)

    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    headers = call_args.kwargs["headers"]

    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer mock-jwt-token-12345"
    assert "encryption-key" not in headers
    assert "AnyLLM-Challenge-Response" not in headers


@pytest.mark.asyncio
async def test_usage_event_includes_version_header(
    mock_platform_client: Mock,
    any_llm_key: str,
    mock_completion: ChatCompletion,
) -> None:
    """Test that usage events include library version in User-Agent header."""
    from any_llm import __version__

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_http_client.post = AsyncMock(return_value=mock_response)

    await post_completion_usage_event(
        platform_client=mock_platform_client,
        client=mock_http_client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
        provider_key_id="550e8400-e29b-41d4-a716-446655440000",
    )

    mock_http_client.post.assert_called_once()
    call_args = mock_http_client.post.call_args
    headers = call_args.kwargs["headers"]

    assert "User-Agent" in headers
    assert headers["User-Agent"] == f"python-any-llm/{__version__}"
