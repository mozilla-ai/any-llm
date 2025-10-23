from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import ValidationError

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.openai import OpenaiProvider
from any_llm.providers.platform import PlatformProvider
from any_llm.providers.platform.utils import get_provider_key, post_completion_usage_event
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


@patch("any_llm.providers.platform.platform.get_provider_key")
def test_prepare_creates_provider(mock_get_provider_key: Mock) -> None:
    """Test proper initialization with an API key from get_provider_key."""
    mock_provider_key = "mock-provider-api-key"
    mock_get_provider_key.return_value = mock_provider_key
    any_llm_key = "ANY.v1.kid123.fingerprint456-base64key"

    provider_instance = PlatformProvider(
        api_key=any_llm_key,
    )
    provider_instance.provider = OpenaiProvider

    assert provider_instance.PROVIDER_NAME == "platform"
    assert provider_instance.provider.PROVIDER_NAME == "openai"
    mock_get_provider_key.assert_called_once_with(
        any_llm_key=any_llm_key,
        provider=OpenaiProvider,
    )


@patch("any_llm.providers.platform.platform.get_provider_key")
def test_prepare_creates_provider_without_api_key(mock_get_provider_key: Mock) -> None:
    """Test error handling when instantiating a PlatformProvider without an ANY_LLM_KEY set."""
    mock_provider_key = "mock-provider-api-key"
    mock_get_provider_key.return_value = mock_provider_key

    with pytest.raises(MissingApiKeyError):
        PlatformProvider()


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.get_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_acompletion_non_streaming_success(
    mock_post_usage: AsyncMock,
    mock_get_provider_key: Mock,
) -> None:
    """Test that non-streaming completions correctly call the provider and post usage events."""
    mock_provider_key = "mock-provider-api-key"
    mock_get_provider_key.return_value = mock_provider_key
    any_llm_key = "ANY.v1.kid123.fingerprint456-base64key"

    mock_completion = ChatCompletion(
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

    provider_instance = PlatformProvider(
        api_key=any_llm_key,
    )
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
    mock_post_usage.assert_called_once_with(
        client=provider_instance.client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=mock_completion,
    )


@pytest.mark.asyncio
@patch("any_llm.providers.platform.platform.get_provider_key")
@patch("any_llm.providers.platform.platform.post_completion_usage_event")
async def test_acompletion_streaming_success(
    mock_post_usage: AsyncMock,
    mock_get_provider_key: Mock,
) -> None:
    """Test that streaming completions correctly wrap the iterator and track usage."""
    mock_provider_key = "mock-provider-api-key"
    mock_get_provider_key.return_value = mock_provider_key
    any_llm_key = "ANY.v1.kid123.fingerprint456-base64key"

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


@patch("any_llm.providers.platform.utils.requests.post")
@patch("any_llm.providers.platform.utils.requests.get")
@patch("any_llm.providers.platform.utils._solve_challenge")
def test_get_provider_key_success(
    mock_solve_challenge: Mock,
    mock_get: Mock,
    mock_post: Mock,
) -> None:
    """Test successful provider key retrieval flow."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    solved_challenge_uuid = "550e8400-e29b-41d4-a716-446655440000"
    encrypted_provider_key = "mock-encrypted-provider-key"
    decrypted_provider_key = "mock-decrypted-provider-key"

    # Mock challenge creation response
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "encrypted_challenge": "mock-encrypted-challenge",
    }

    # Mock challenge solving
    mock_solve_challenge.return_value = solved_challenge_uuid

    # Mock provider key fetch response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "encrypted_key": encrypted_provider_key,
    }

    # Mock decryption by patching _decrypt_provider_key
    with patch("any_llm.providers.platform.utils._decrypt_provider_key") as mock_decrypt:
        mock_decrypt.return_value = decrypted_provider_key

        # Call the function
        result = get_provider_key(any_llm_key=any_llm_key, provider=OpenaiProvider)

        # Assertions
        assert result == decrypted_provider_key
        mock_post.assert_called_once()
        mock_get.assert_called_once()
        mock_solve_challenge.assert_called_once()
        mock_decrypt.assert_called_once()


@patch("any_llm.providers.platform.utils.requests.post")
def test_get_provider_key_invalid_api_key_format(mock_post: Mock) -> None:
    """Test error handling when ANY_LLM_KEY has invalid format."""
    invalid_key = "INVALID_KEY_FORMAT"

    with pytest.raises(ValueError, match="Invalid ANY_API_KEY format"):
        get_provider_key(any_llm_key=invalid_key, provider=OpenaiProvider)

    mock_post.assert_not_called()


@patch("any_llm.providers.platform.utils.requests.post")
def test_get_provider_key_challenge_creation_failure(mock_post: Mock) -> None:
    """Test error handling when challenge creation fails."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="

    # Mock failed challenge creation
    mock_post.return_value.status_code = 400
    mock_post.return_value.json.return_value = {"error": "Bad request"}
    mock_post.return_value.text = "Bad request"

    with pytest.raises(RuntimeError, match="Bad request"):
        get_provider_key(any_llm_key=any_llm_key, provider=OpenaiProvider)

    mock_post.assert_called_once()


@patch("any_llm.providers.platform.utils.requests.post")
@patch("any_llm.providers.platform.utils.requests.get")
@patch("any_llm.providers.platform.utils._solve_challenge")
def test_get_provider_key_fetch_failure(
    mock_solve_challenge: Mock,
    mock_get: Mock,
    mock_post: Mock,
) -> None:
    """Test error handling when provider key fetch fails."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    solved_challenge_uuid = "550e8400-e29b-41d4-a716-446655440000"

    # Mock successful challenge creation
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "encrypted_challenge": "mock-encrypted-challenge",
    }

    # Mock challenge solving
    mock_solve_challenge.return_value = solved_challenge_uuid

    # Mock failed provider key fetch
    mock_get.return_value.status_code = 404
    mock_get.return_value.json.return_value = {"error": "Provider key not found"}
    mock_get.return_value.text = "Provider key not found"

    with pytest.raises(RuntimeError, match="Provider key not found"):
        get_provider_key(any_llm_key=any_llm_key, provider=OpenaiProvider)

    mock_post.assert_called_once()
    mock_get.assert_called_once()
    mock_solve_challenge.assert_called_once()


@pytest.mark.asyncio
@patch("any_llm.providers.platform.utils.requests.post")
@patch("any_llm.providers.platform.utils.requests.get")
@patch("any_llm.providers.platform.utils._solve_challenge")
async def test_post_completion_usage_event_success(
    mock_solve_challenge: Mock,
    mock_get: Mock,
    mock_post: Mock,
) -> None:
    """Test successful posting of completion usage event."""
    any_llm_key = "ANY.v1.kid123.fingerprint456-YWJjZGVmZ2hpamtsbW5vcHFyc3R1dnd4eXoxMjM0NTY="
    solved_challenge_uuid = "550e8400-e29b-41d4-a716-446655440000"
    provider_key_id = "provider-key-123"

    # Create mock completion
    completion = ChatCompletion(
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

    # Mock challenge creation response (called twice)
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "encrypted_challenge": "mock-encrypted-challenge",
    }

    # Mock challenge solving (called twice)
    mock_solve_challenge.return_value = solved_challenge_uuid

    # Mock provider key fetch response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "id": provider_key_id,
        "encrypted_key": "mock-encrypted-key",
    }

    # Create mock httpx client
    mock_response = AsyncMock()
    mock_response.raise_for_status = AsyncMock()

    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(return_value=mock_response)

    # Call the function
    await post_completion_usage_event(
        client=client,
        any_llm_key=any_llm_key,
        provider="openai",
        completion=completion,
    )

    # Assertions
    # Challenge creation should be called twice (once for provider key, once for usage event)
    assert mock_post.call_count == 2

    # Provider key fetch should be called once
    mock_get.assert_called_once()

    # Challenge solving should be called twice
    assert mock_solve_challenge.call_count == 2

    # Usage event POST should be called once
    client.post.assert_called_once()

    # Verify the payload sent to the usage event endpoint
    call_args = client.post.call_args
    assert "/usage-events/" in call_args.args[0]
    payload = call_args.kwargs["json"]
    assert payload["provider_key_id"] == provider_key_id
    assert payload["provider"] == "openai"
    assert payload["model"] == "gpt-4"
    assert payload["data"]["input_tokens"] == "10"
    assert payload["data"]["output_tokens"] == "5"
    assert "id" in payload


@pytest.mark.asyncio
@patch("any_llm.providers.platform.utils.requests.post")
async def test_post_completion_usage_event_invalid_key_format(mock_post: Mock) -> None:
    """Test error handling when ANY_LLM_KEY has invalid format."""
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

    client = AsyncMock(spec=httpx.AsyncClient)

    with pytest.raises(ValueError, match="Invalid ANY_API_KEY format"):
        await post_completion_usage_event(
            client=client,
            any_llm_key=invalid_key,
            provider="openai",
            completion=completion,
        )

    mock_post.assert_not_called()


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
    mock_platform_class.assert_called_once_with(api_key=any_llm_key, api_base=None)


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
