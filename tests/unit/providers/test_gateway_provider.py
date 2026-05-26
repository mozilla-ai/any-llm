from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("otari")

from any_llm.providers.gateway import GatewayProvider
from any_llm.providers.otari import OtariProvider


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


@patch.dict("os.environ", {"OTARI_API_BASE": "https://otari.env"}, clear=False)
def test_otari_provider_prefers_otari_env_api_base() -> None:
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()
        OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_base"] == "https://otari.env"


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
