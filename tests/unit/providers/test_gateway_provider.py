import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from any_llm.exceptions import (
    AuthenticationError,
    GatewayTimeoutError,
    InsufficientFundsError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.providers.gateway.gateway import (
    GATEWAY_HEADER_NAME,
    GATEWAY_PLATFORM_TOKEN_ENV,
    GatewayProvider,
)

# -- Non-platform mode (existing behaviour) -----------------------------------


def test_gateway_init_requires_api_base() -> None:
    with pytest.raises(ValueError, match="api_base is required"):
        GatewayProvider(api_key="test-key")


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_init_with_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")

    mock_openai_class.assert_called_once()
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["base_url"] == "https://gateway.example.com"
    assert call_kwargs["api_key"] == "test-key"
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer test-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_init_without_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_base="https://gateway.example.com")

    mock_openai_class.assert_called_once()
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["base_url"] == "https://gateway.example.com"
    assert call_kwargs["api_key"] == ""


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch("any_llm.providers.gateway.gateway.logger")
def test_gateway_init_header_override_warning(mock_logger: MagicMock, mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(
        api_key="new-key",
        api_base="https://gateway.example.com",
        default_headers={GATEWAY_HEADER_NAME: "Bearer old-key"},
    )

    mock_logger.info.assert_called_once()
    assert "already set" in mock_logger.info.call_args[0][0]
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer new-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {"GATEWAY_API_KEY": "env-key"}, clear=False)
def test_gateway_init_with_env_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_base="https://gateway.example.com")

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == "env-key"
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer env-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {}, clear=True)
def test_gateway_init_without_any_api_key(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(api_base="https://gateway.example.com")

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == ""
    assert "default_headers" not in call_kwargs or GATEWAY_HEADER_NAME not in call_kwargs.get("default_headers", {})


def test_verify_api_key_with_provided_key() -> None:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")
        result = provider._verify_and_set_api_key("provided-key")
        assert result == "provided-key"


@patch.dict(os.environ, {"GATEWAY_API_KEY": "env-key"}, clear=False)
def test_verify_api_key_with_env_variable() -> None:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")
        result = provider._verify_and_set_api_key(None)
        assert result == "env-key"


@patch.dict(os.environ, {}, clear=True)
def test_verify_api_key_none_returns_empty() -> None:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_base="https://gateway.example.com")
        result = provider._verify_and_set_api_key(None)
        assert result == ""


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_client_initialization_with_custom_headers(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    custom_headers = {"X-Custom-Header": "custom-value"}
    GatewayProvider(api_key="test-key", api_base="https://gateway.example.com", default_headers=custom_headers)

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer test-key"
    assert call_kwargs["default_headers"]["X-Custom-Header"] == "custom-value"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_passes_kwargs_to_parent(mock_openai_class: MagicMock) -> None:
    mock_client = AsyncMock()
    mock_openai_class.return_value = mock_client

    GatewayProvider(
        api_key="test-key",
        api_base="https://gateway.example.com",
        timeout=30,
        max_retries=5,
        default_headers={},
    )

    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["timeout"] == 30
    assert call_kwargs["max_retries"] == 5
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer test-key"


# -- Platform mode: initialisation -------------------------------------------


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_platform_mode_explicit(mock_openai_class: MagicMock) -> None:
    """Explicit platform_mode=True sends only Authorization header, not X-AnyLLM-Key."""
    mock_openai_class.return_value = AsyncMock()

    provider = GatewayProvider(
        api_key="user-token",
        api_base="https://gateway.example.com",
        platform_mode=True,
    )

    assert provider.platform_mode is True
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == "user-token"
    assert call_kwargs["base_url"] == "https://gateway.example.com"
    assert GATEWAY_HEADER_NAME not in call_kwargs.get("default_headers", {})


def test_gateway_platform_mode_requires_token() -> None:
    """platform_mode=True without a token raises ValueError."""
    with pytest.raises(ValueError, match="Platform mode requires a user token"):
        GatewayProvider(api_base="https://gateway.example.com", platform_mode=True)


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {GATEWAY_PLATFORM_TOKEN_ENV: "env-platform-token"}, clear=False)
def test_gateway_platform_mode_explicit_with_env_token(mock_openai_class: MagicMock) -> None:
    """platform_mode=True falls back to GATEWAY_PLATFORM_TOKEN env var."""
    mock_openai_class.return_value = AsyncMock()

    provider = GatewayProvider(api_base="https://gateway.example.com", platform_mode=True)

    assert provider.platform_mode is True
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == "env-platform-token"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {GATEWAY_PLATFORM_TOKEN_ENV: "env-platform-token"}, clear=False)
def test_gateway_platform_mode_auto_detect_via_env(mock_openai_class: MagicMock) -> None:
    """When GATEWAY_PLATFORM_TOKEN is set and no api_key, auto-enter platform mode."""
    mock_openai_class.return_value = AsyncMock()

    provider = GatewayProvider(api_base="https://gateway.example.com")

    assert provider.platform_mode is True
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == "env-platform-token"
    assert GATEWAY_HEADER_NAME not in call_kwargs.get("default_headers", {})


@patch("any_llm.providers.openai.base.AsyncOpenAI")
@patch.dict(os.environ, {GATEWAY_PLATFORM_TOKEN_ENV: "env-platform-token"}, clear=False)
def test_gateway_platform_mode_env_ignored_when_api_key_provided(mock_openai_class: MagicMock) -> None:
    """When an explicit api_key is provided, GATEWAY_PLATFORM_TOKEN does not trigger platform mode."""
    mock_openai_class.return_value = AsyncMock()

    provider = GatewayProvider(api_key="gateway-key", api_base="https://gateway.example.com")

    assert provider.platform_mode is False
    call_kwargs = mock_openai_class.call_args[1]
    assert call_kwargs["api_key"] == "gateway-key"
    assert call_kwargs["default_headers"][GATEWAY_HEADER_NAME] == "Bearer gateway-key"


@patch("any_llm.providers.openai.base.AsyncOpenAI")
def test_gateway_default_mode_sets_platform_mode_false(mock_openai_class: MagicMock) -> None:
    """Non-platform init sets platform_mode to False."""
    mock_openai_class.return_value = AsyncMock()

    provider = GatewayProvider(api_key="test-key", api_base="https://gateway.example.com")
    assert provider.platform_mode is False


# -- Platform mode: error handling --------------------------------------------


def _make_api_status_error(
    status_code: int,
    message: str = "error",
    headers: dict[str, str] | None = None,
) -> openai.APIStatusError:
    """Build an ``openai.APIStatusError`` with a fake httpx response."""
    resp_headers = {"content-type": "application/json"}
    if headers:
        resp_headers.update(headers)
    response = httpx.Response(
        status_code=status_code,
        headers=resp_headers,
        json={"error": {"message": message}},
        request=httpx.Request("POST", "https://gateway.example.com/v1/chat/completions"),
    )
    return openai.APIStatusError(
        message=message,
        response=response,
        body={"error": {"message": message}},
    )


def _make_platform_provider() -> GatewayProvider:
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        return GatewayProvider(
            api_key="user-token",
            api_base="https://gateway.example.com",
            platform_mode=True,
        )


def test_gateway_platform_error_401() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(401, "Invalid token")

    with pytest.raises(AuthenticationError, match="Invalid token"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_402() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(402, "Budget exceeded")

    with pytest.raises(InsufficientFundsError, match="Budget exceeded"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_403() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(403, "Forbidden")

    with pytest.raises(AuthenticationError, match="Forbidden"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_404() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(404, "Model not found")

    with pytest.raises(ModelNotFoundError, match="Model not found"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_429_with_retry_after() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(429, "Too many requests", headers={"retry-after": "30"})

    with pytest.raises(RateLimitError) as exc_info:
        provider._handle_platform_error(exc)

    assert exc_info.value.retry_after == "30"
    assert "Too many requests" in str(exc_info.value)


def test_gateway_platform_error_429_without_retry_after() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(429, "Rate limited")

    with pytest.raises(RateLimitError) as exc_info:
        provider._handle_platform_error(exc)

    assert exc_info.value.retry_after is None


def test_gateway_platform_error_502() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(502, "Bad gateway")

    with pytest.raises(UpstreamProviderError, match="Bad gateway"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_504() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(504, "Gateway timeout")

    with pytest.raises(GatewayTimeoutError, match="Gateway timeout"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_correlation_id() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(
        500,
        "Internal error",
        headers={"x-correlation-id": "abc-123"},
    )

    with pytest.raises(openai.APIStatusError):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_correlation_id_in_mapped_error() -> None:
    provider = _make_platform_provider()
    exc = _make_api_status_error(
        401,
        "Unauthorized",
        headers={"x-correlation-id": "trace-xyz"},
    )

    with pytest.raises(AuthenticationError, match="correlation_id=trace-xyz"):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_unknown_status_reraises() -> None:
    """Unrecognised status codes pass through unchanged."""
    provider = _make_platform_provider()
    exc = _make_api_status_error(500, "Internal server error")

    with pytest.raises(openai.APIStatusError):
        provider._handle_platform_error(exc)


def test_gateway_platform_error_non_api_error_reraises() -> None:
    """Non-APIStatusError exceptions pass through unchanged."""
    provider = _make_platform_provider()

    with pytest.raises(RuntimeError, match="something else"):
        provider._handle_platform_error(RuntimeError("something else"))


@pytest.mark.asyncio
async def test_gateway_platform_acompletion_wraps_errors() -> None:
    """_acompletion wraps APIStatusError in platform mode."""
    provider = _make_platform_provider()
    exc = _make_api_status_error(429, "Rate limited", headers={"retry-after": "5"})
    provider.client = AsyncMock()
    provider.client.chat.completions.create = AsyncMock(side_effect=exc)

    with pytest.raises(RateLimitError) as exc_info:
        await provider._acompletion(
            MagicMock(
                model_id="openai:gpt-4",
                messages=[],
                reasoning_effort=None,
                stream=False,
                response_format=None,
            ),
        )

    assert exc_info.value.retry_after == "5"


@pytest.mark.asyncio
async def test_gateway_non_platform_acompletion_no_wrapping() -> None:
    """In non-platform mode, errors are not wrapped by _handle_platform_error."""
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = GatewayProvider(api_key="key", api_base="https://gateway.example.com")

    assert provider.platform_mode is False
    exc = _make_api_status_error(429, "Rate limited")
    provider.client = AsyncMock()
    provider.client.chat.completions.create = AsyncMock(side_effect=exc)

    with pytest.raises(openai.APIStatusError):
        await provider._acompletion(
            MagicMock(
                model_id="openai:gpt-4",
                messages=[],
                reasoning_effort=None,
                stream=False,
                response_format=None,
            ),
        )
