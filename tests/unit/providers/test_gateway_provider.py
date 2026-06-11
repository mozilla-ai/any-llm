from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("otari")

from otari import errors as otari_errors

from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    GatewayTimeoutError,
    InsufficientFundsError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.providers.gateway import GatewayProvider
from any_llm.providers.otari import OtariProvider
from any_llm.providers.otari.otari import LEGACY_GATEWAY_HEADER_NAME
from any_llm.types.completion import CompletionParams


def _mock_otari_client(platform_mode: bool = False) -> MagicMock:
    mock_client = MagicMock()
    mock_client.platform_mode = platform_mode
    mock_client.completion = AsyncMock()
    return mock_client


def _build_gateway_provider() -> GatewayProvider:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient", return_value=_mock_otari_client(platform_mode=True)):
        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            return GatewayProvider(api_key="platform-token", api_base="https://otari.example.com", platform_mode=True)


def test_gateway_provider_warns_and_uses_otari_provider() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()

        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            provider = GatewayProvider(api_base="https://otari.example.com")

    assert isinstance(provider, OtariProvider)


def test_gateway_provider_is_backed_by_otari_client() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mocked_client = _mock_otari_client(platform_mode=True)
        mock_otari_client.return_value = mocked_client

        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            provider = GatewayProvider(api_base="https://otari.example.com")

    assert provider.platform_mode is True
    assert provider.otari_client is mocked_client
    # otari 0.1.0 has no embedded OpenAI client; self.client points at the otari client itself.
    assert provider.client is mocked_client


def test_gateway_provider_uses_gateway_identity_and_env_names() -> None:
    assert GatewayProvider.PROVIDER_NAME == "gateway"
    assert GatewayProvider.ENV_API_KEY_NAME == "GATEWAY_API_KEY"
    assert GatewayProvider.ENV_API_BASE_NAME == "GATEWAY_API_BASE"


def test_gateway_header_constant_is_legacy_name() -> None:
    from any_llm.providers.gateway.gateway import GATEWAY_HEADER_NAME

    assert GATEWAY_HEADER_NAME == LEGACY_GATEWAY_HEADER_NAME


@patch.dict(
    "os.environ", {"GATEWAY_API_BASE": "https://gateway.env", "OTARI_API_BASE": "https://otari.env"}, clear=False
)
def test_gateway_provider_prefers_gateway_env_api_base() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()

        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            GatewayProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_base"] == "https://gateway.env"


@patch.dict("os.environ", {"OTARI_API_BASE": "https://otari.env"}, clear=False)
def test_otari_provider_prefers_otari_env_api_base() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()
        OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_base"] == "https://otari.env"


@patch.dict(
    "os.environ",
    {
        "OTARI_API_BASE": "https://otari.env",
        "OTARI_PLATFORM_TOKEN": "platform-token",
        "OTARI_API_KEY": "",
    },
    clear=False,
)
def test_otari_provider_platform_mode_uses_platform_token_not_placeholder_key() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client(platform_mode=True)
        OtariProvider(platform_mode=True)

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["platform_token"] == "platform-token"  # noqa: S105
    assert "api_key" not in call_kwargs


@patch.dict(
    "os.environ",
    {
        "OTARI_API_BASE": "https://otari.env",
        "OTARI_PLATFORM_TOKEN": "platform-token",
        "OTARI_API_KEY": "",
    },
    clear=False,
)
def test_otari_provider_auto_mode_prefers_platform_token_over_placeholder_key() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client(platform_mode=True)
        OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["platform_token"] == "platform-token"  # noqa: S105
    assert "api_key" not in call_kwargs


@patch.dict("os.environ", {"GATEWAY_API_BASE": "https://gateway.env"}, clear=False)
def test_otari_provider_falls_back_to_gateway_env_api_base() -> None:
    with patch.dict("os.environ", {"OTARI_API_BASE": ""}, clear=False):
        with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
            mock_otari_client.return_value = _mock_otari_client()
            OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_base"] == "https://gateway.env"


def test_otari_provider_requires_api_base_when_no_env() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="api_base is required"):
            OtariProvider()


@pytest.mark.asyncio
async def test_otari_completion_sends_max_tokens() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.completion = AsyncMock(
        return_value={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    with patch("any_llm.providers.otari.otari.AsyncOtariClient", return_value=mocked_client):
        provider = OtariProvider(api_base="https://otari.example.com")
    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
    )

    await provider._acompletion(params)

    call_kwargs = mocked_client.completion.call_args.kwargs
    assert call_kwargs["max_tokens"] == 64
    assert "max_completion_tokens" not in call_kwargs


@pytest.mark.parametrize(
    ("otari_exc", "expected"),
    [
        (otari_errors.AuthenticationError("nope"), AuthenticationError),
        (otari_errors.ModelNotFoundError("missing"), ModelNotFoundError),
        (otari_errors.InsufficientFundsError("broke"), InsufficientFundsError),
        (otari_errors.UpstreamProviderError("upstream"), UpstreamProviderError),
        (otari_errors.GatewayTimeoutError("slow"), GatewayTimeoutError),
    ],
)
def test_gateway_handle_platform_error_maps_otari_exceptions(otari_exc: Exception, expected: type[Exception]) -> None:
    provider = _build_gateway_provider()
    with pytest.raises(expected) as exc_info:
        provider._handle_platform_error(otari_exc)
    err = exc_info.value
    assert isinstance(err, AnyLLMError)
    assert err.__cause__ is otari_exc
    assert err.provider_name == "gateway"


def test_gateway_handle_platform_error_maps_rate_limit_with_retry_after() -> None:
    provider = _build_gateway_provider()
    otari_exc = otari_errors.RateLimitError("slow down", retry_after="5")
    with pytest.raises(RateLimitError) as exc_info:
        provider._handle_platform_error(otari_exc)
    assert exc_info.value.retry_after == "5"
    assert exc_info.value.__cause__ is otari_exc


def test_gateway_handle_platform_error_reraises_non_otari_error() -> None:
    provider = _build_gateway_provider()
    sentinel = ValueError("not an otari error")
    with pytest.raises(ValueError, match="not an otari error"):
        provider._handle_platform_error(sentinel)


def test_gateway_handle_platform_error_reraises_unmapped_otari_error() -> None:
    provider = _build_gateway_provider()
    base_exc = otari_errors.OtariError("generic")
    with pytest.raises(otari_errors.OtariError, match="generic"):
        provider._handle_platform_error(base_exc)


@pytest.mark.asyncio
async def test_gateway_moderation_maps_platform_error() -> None:
    mocked_client = _mock_otari_client(platform_mode=True)
    mocked_client.moderation = AsyncMock(side_effect=otari_errors.AuthenticationError("bad token"))
    with patch("any_llm.providers.otari.otari.AsyncOtariClient", return_value=mocked_client):
        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            provider = GatewayProvider(
                api_key="platform-token", api_base="https://otari.example.com", platform_mode=True
            )

    with pytest.raises(AuthenticationError):
        await provider._amoderation(model="omni-moderation-latest", input="hello")
