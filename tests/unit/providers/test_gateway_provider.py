from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("otari")

from any_llm.providers.gateway import GatewayProvider
from any_llm.providers.otari import OtariProvider
from any_llm.providers.otari.otari import LEGACY_GATEWAY_HEADER_NAME
from any_llm.types.completion import CompletionParams


def _mock_otari_client(platform_mode: bool = False) -> MagicMock:
    mock_client = MagicMock()
    mock_client.platform_mode = platform_mode
    mock_client.openai = AsyncMock()
    return mock_client


def test_gateway_provider_warns_and_uses_otari_provider() -> None:
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()

        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            provider = GatewayProvider(api_base="https://otari.example.com")

    assert isinstance(provider, OtariProvider)


def test_gateway_provider_is_backed_by_otari_client() -> None:
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
        mocked_client = _mock_otari_client(platform_mode=True)
        mock_otari_client.return_value = mocked_client

        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            provider = GatewayProvider(api_base="https://otari.example.com")

    assert provider.platform_mode is True
    assert provider.otari_client is mocked_client
    assert provider.client is mocked_client.openai


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
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()

        with pytest.warns(DeprecationWarning, match="gateway.*deprecated"):
            GatewayProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_base"] == "https://gateway.env"


@patch.dict("os.environ", {"OTARI_API_BASE": "https://otari.env"}, clear=False)
def test_otari_provider_prefers_otari_env_api_base() -> None:
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
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
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
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
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client(platform_mode=True)
        OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["platform_token"] == "platform-token"  # noqa: S105
    assert "api_key" not in call_kwargs


@patch.dict("os.environ", {"GATEWAY_API_BASE": "https://gateway.env"}, clear=False)
def test_otari_provider_falls_back_to_gateway_env_api_base() -> None:
    with patch.dict("os.environ", {"OTARI_API_BASE": ""}, clear=False):
        with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
            mock_otari_client.return_value = _mock_otari_client()
            OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_base"] == "https://gateway.env"


def test_otari_provider_requires_api_base_when_no_env() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="api_base is required"):
            OtariProvider()


@pytest.mark.asyncio
async def test_otari_completion_converts_max_tokens_to_max_completion_tokens() -> None:
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

    with patch("any_llm.providers.otari.otari.OtariClient", return_value=mocked_client):
        provider = OtariProvider(api_base="https://otari.example.com")
    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
    )

    await provider._acompletion(params)

    call_kwargs = mocked_client.completion.call_args.kwargs
    assert call_kwargs["max_completion_tokens"] == 64
    assert "max_tokens" not in call_kwargs
