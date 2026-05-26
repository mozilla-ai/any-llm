from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.providers.github.github import GithubProvider
from any_llm.types.completion import CompletionParams


def test_github_provider_attributes() -> None:
    provider = GithubProvider(api_key="test-key")
    assert provider.PROVIDER_NAME == "github"
    assert provider.ENV_API_KEY_NAME == "GITHUB_TOKEN"
    assert provider.ENV_API_BASE_NAME == "GITHUB_MODELS_API_BASE"
    assert provider.API_BASE == "https://models.github.ai/inference"
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_EMBEDDING is True
    assert provider.SUPPORTS_LIST_MODELS is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is False
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_COMPLETION_REASONING is False
    assert provider.SUPPORTS_RESPONSES is False
    assert provider.SUPPORTS_MODERATION is False
    assert provider.SUPPORTS_BATCH is False


def test_github_max_tokens_stays_as_max_tokens_in_payload() -> None:
    """max_tokens in CompletionParams flows through as max_tokens in the final payload.

    BaseOpenAIProvider first remaps max_tokens to max_completion_tokens, then
    GithubProvider remaps it back to max_tokens because the GitHub API only
    accepts max_tokens.
    """
    params = CompletionParams(
        model_id="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    result = GithubProvider._convert_completion_params(params)
    assert "max_completion_tokens" not in result
    assert result["max_tokens"] == 1024


def test_github_no_max_tokens_produces_no_max_token_fields() -> None:
    """When no max_tokens is provided, neither max_tokens nor max_completion_tokens appear."""
    params = CompletionParams(
        model_id="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "Hello"}],
    )
    result = GithubProvider._convert_completion_params(params)
    assert "max_completion_tokens" not in result
    assert "max_tokens" not in result


def test_github_convert_list_models_response() -> None:
    catalog_data = [
        {
            "id": "openai/gpt-4.1",
            "name": "GPT-4.1",
            "registry": "azure-openai",
            "publisher": "OpenAI",
            "summary": "A large language model",
            "rate_limit_tier": "high",
            "capabilities": ["chat"],
            "limits": {"max_input_tokens": 128000, "max_output_tokens": 16384},
            "supported_input_modalities": ["text"],
            "supported_output_modalities": ["text"],
        },
        {
            "id": "meta/llama-3-70b",
            "name": "Llama 3 70B",
            "registry": "azure",
            "publisher": "Meta",
            "summary": "An open-weight model",
            "rate_limit_tier": "low",
            "capabilities": ["chat"],
            "limits": {"max_input_tokens": 8192, "max_output_tokens": 4096},
            "supported_input_modalities": ["text"],
            "supported_output_modalities": ["text"],
        },
    ]
    models = GithubProvider._convert_list_models_response(catalog_data)
    assert len(models) == 2
    assert models[0].id == "openai/gpt-4.1"
    assert models[0].owned_by == "OpenAI"
    assert models[0].object == "model"
    assert models[1].id == "meta/llama-3-70b"
    assert models[1].owned_by == "Meta"


def test_github_convert_list_models_response_missing_publisher() -> None:
    catalog_data = [{"id": "some/model"}]
    models = GithubProvider._convert_list_models_response(catalog_data)
    assert len(models) == 1
    assert models[0].owned_by == "unknown"


def test_github_convert_list_models_response_empty() -> None:
    models = GithubProvider._convert_list_models_response([])
    assert models == []


def test_github_convert_list_models_response_non_list() -> None:
    models = GithubProvider._convert_list_models_response("unexpected")
    assert models == []


def test_github_convert_list_models_response_skips_non_dict_entries() -> None:
    catalog_data = [
        {"id": "openai/gpt-4.1", "publisher": "OpenAI"},
        "not-a-dict",
        42,
        {"id": "meta/llama-3-70b", "publisher": "Meta"},
    ]
    models = GithubProvider._convert_list_models_response(catalog_data)
    assert len(models) == 2
    assert models[0].id == "openai/gpt-4.1"
    assert models[1].id == "meta/llama-3-70b"


def test_github_convert_list_models_response_skips_entries_missing_id() -> None:
    catalog_data = [
        {"id": "openai/gpt-4.1", "publisher": "OpenAI"},
        {"name": "No ID model", "publisher": "Unknown"},
        {"id": "", "publisher": "Empty"},
    ]
    models = GithubProvider._convert_list_models_response(catalog_data)
    assert len(models) == 1
    assert models[0].id == "openai/gpt-4.1"


def test_github_catalog_url_derived_from_default_base() -> None:
    provider = GithubProvider(api_key="test-key")
    assert provider._get_catalog_url() == "https://models.github.ai/catalog/models"


def test_github_catalog_url_derived_from_custom_base() -> None:
    provider = GithubProvider(api_key="test-key", api_base="https://custom.host/inference")
    assert provider._get_catalog_url() == "https://custom.host/catalog/models"


def test_github_model_id_split() -> None:
    provider_enum, model_name = AnyLLM.split_model_provider("github:openai/gpt-4.1")
    assert provider_enum == LLMProvider.GITHUB
    assert model_name == "openai/gpt-4.1"


@pytest.mark.asyncio
async def test_github_alist_models() -> None:
    catalog_response = [
        {
            "id": "openai/gpt-4.1-nano",
            "name": "GPT-4.1 Nano",
            "publisher": "OpenAI",
        },
    ]
    mock_response = Mock()
    mock_response.json.return_value = catalog_response
    mock_response.raise_for_status = Mock()

    provider = GithubProvider(api_key="test-token")

    with patch("any_llm.providers.github.github.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        models = await provider._alist_models()

        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://models.github.ai/catalog/models"
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-token"
        assert headers["Accept"] == "application/vnd.github+json"
        assert call_args[1]["timeout"] == 60.0

        assert len(models) == 1
        assert models[0].id == "openai/gpt-4.1-nano"
        assert models[0].owned_by == "OpenAI"


@pytest.mark.asyncio
async def test_github_alist_models_raises_on_http_error() -> None:
    provider = GithubProvider(api_key="bad-token")

    mock_response = Mock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=httpx.Request("GET", "https://models.github.ai/catalog/models"),
        response=httpx.Response(401),
    )

    with patch("any_llm.providers.github.github.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_class.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            await provider._alist_models()
