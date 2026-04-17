from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from any_llm.types.rerank import RerankMeta, RerankResponse, RerankResult, RerankUsage


def test_rerank_response_model() -> None:
    resp = RerankResponse(
        id="test-id",
        results=[
            RerankResult(index=0, relevance_score=0.9),
            RerankResult(index=1, relevance_score=0.5),
        ],
        usage=RerankUsage(total_tokens=100),
    )
    assert resp.id == "test-id"
    assert len(resp.results) == 2
    assert resp.results[0].relevance_score == 0.9
    assert resp.usage is not None
    assert resp.usage.total_tokens == 100
    assert resp.meta is None


def test_rerank_response_optional_fields() -> None:
    resp = RerankResponse(id="test", results=[])
    assert resp.meta is None
    assert resp.usage is None


def test_rerank_meta_model() -> None:
    meta = RerankMeta(
        billed_units={"search_units": 1.0},
        tokens={"input_tokens": 50},
    )
    assert meta.billed_units is not None
    assert meta.billed_units["search_units"] == 1.0
    assert meta.tokens is not None
    assert meta.tokens["input_tokens"] == 50


def test_rerank_meta_optional_fields() -> None:
    meta = RerankMeta()
    assert meta.billed_units is None
    assert meta.tokens is None


def test_rerank_result_model() -> None:
    result = RerankResult(index=2, relevance_score=0.85)
    assert result.index == 2
    assert result.relevance_score == 0.85


def test_rerank_usage_model() -> None:
    usage = RerankUsage(total_tokens=42)
    assert usage.total_tokens == 42


def test_rerank_usage_optional() -> None:
    usage = RerankUsage()
    assert usage.total_tokens is None


def test_cohere_supports_rerank() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    assert CohereProvider.SUPPORTS_RERANK is True


def test_gateway_supports_rerank() -> None:
    from any_llm.providers.gateway import GatewayProvider

    assert GatewayProvider.SUPPORTS_RERANK is True


def test_openai_base_does_not_support_rerank() -> None:
    from any_llm.providers.openai.base import BaseOpenAIProvider

    assert BaseOpenAIProvider.SUPPORTS_RERANK is False


def test_anthropic_base_does_not_support_rerank() -> None:
    from any_llm.providers.anthropic.base import BaseAnthropicProvider

    assert BaseAnthropicProvider.SUPPORTS_RERANK is False


def test_cohere_convert_rerank_params_basic() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    params = CohereProvider._convert_rerank_params(
        model="rerank-v3.5",
        query="test query",
        documents=["doc1", "doc2"],
    )
    assert params["query"] == "test query"
    assert params["documents"] == ["doc1", "doc2"]
    assert "top_n" not in params
    assert "max_tokens_per_doc" not in params


def test_cohere_convert_rerank_params_with_options() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    params = CohereProvider._convert_rerank_params(
        model="rerank-v3.5",
        query="test query",
        documents=["doc1", "doc2"],
        top_n=2,
        max_tokens_per_doc=512,
        return_documents=True,
    )
    assert params["query"] == "test query"
    assert params["documents"] == ["doc1", "doc2"]
    assert params["top_n"] == 2
    assert params["max_tokens_per_doc"] == 512
    assert params["return_documents"] is True


def test_cohere_convert_rerank_params_ignores_none_top_n() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    params = CohereProvider._convert_rerank_params(
        model="rerank-v3.5",
        query="q",
        documents=["d"],
        top_n=None,
    )
    assert "top_n" not in params


def test_convert_cohere_rerank_response_basic() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.utils import _convert_cohere_rerank_response

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.95

    mock_response = MagicMock()
    mock_response.id = "rerank-123"
    mock_response.results = [mock_result]
    mock_response.meta = None

    result = _convert_cohere_rerank_response(mock_response)
    assert isinstance(result, RerankResponse)
    assert result.id == "rerank-123"
    assert len(result.results) == 1
    assert result.results[0].relevance_score == 0.95
    assert result.meta is None
    assert result.usage is None


def test_convert_cohere_rerank_response_sorts_descending() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.utils import _convert_cohere_rerank_response

    mock_r1 = MagicMock()
    mock_r1.index = 0
    mock_r1.relevance_score = 0.3

    mock_r2 = MagicMock()
    mock_r2.index = 1
    mock_r2.relevance_score = 0.9

    mock_response = MagicMock()
    mock_response.id = "rerank-456"
    mock_response.results = [mock_r1, mock_r2]
    mock_response.meta = None

    result = _convert_cohere_rerank_response(mock_response)
    assert result.results[0].relevance_score == 0.9
    assert result.results[1].relevance_score == 0.3


def test_convert_cohere_rerank_response_with_meta() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.utils import _convert_cohere_rerank_response

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.8

    mock_meta = MagicMock()
    mock_meta.billed_units = MagicMock()
    mock_meta.billed_units.search_units = 1.0
    mock_meta.tokens = MagicMock()
    mock_meta.tokens.input_tokens = 100

    mock_response = MagicMock()
    mock_response.id = "rerank-789"
    mock_response.results = [mock_result]
    mock_response.meta = mock_meta

    result = _convert_cohere_rerank_response(mock_response)
    assert result.meta is not None
    assert result.meta.billed_units is not None
    assert result.meta.billed_units["search_units"] == 1.0
    assert result.meta.tokens is not None
    assert result.meta.tokens["input_tokens"] == 100
    assert result.usage is not None
    assert result.usage.total_tokens == 100


def test_convert_cohere_rerank_response_with_no_billing() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.utils import _convert_cohere_rerank_response

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.5

    mock_meta = MagicMock()
    mock_meta.billed_units = None
    mock_meta.tokens = None

    mock_response = MagicMock()
    mock_response.id = "rerank-000"
    mock_response.results = [mock_result]
    mock_response.meta = mock_meta

    result = _convert_cohere_rerank_response(mock_response)
    assert result.meta is None
    assert result.usage is None


@pytest.mark.asyncio
async def test_unsupported_provider_raises() -> None:
    from any_llm.providers.openai.openai import OpenaiProvider

    provider = OpenaiProvider(api_key="test-key")
    with pytest.raises(NotImplementedError, match="doesn't support rerank"):
        await provider._arerank("model", "query", ["doc"])


def test_cohere_metadata_includes_rerank() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    meta = CohereProvider.get_provider_metadata()
    assert meta.rerank is True


def test_openai_metadata_rerank_false() -> None:
    from any_llm.providers.openai.openai import OpenaiProvider

    meta = OpenaiProvider.get_provider_metadata()
    assert meta.rerank is False


@pytest.mark.asyncio
async def test_gateway_rerank_http_call() -> None:
    """Verify the gateway provider makes a POST to /v1/rerank."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "rerank-123",
            "results": [{"index": 0, "relevance_score": 0.9}],
            "usage": {"total_tokens": 50},
        }
        mock_client.post.return_value = mock_resp

        from any_llm.providers.gateway import GatewayProvider

        provider = GatewayProvider(api_key="test-key", api_base="http://localhost:8000")
        result = await provider._arerank("rerank-v3.5", "query", ["doc1"])

        assert isinstance(result, RerankResponse)
        assert result.id == "rerank-123"
        assert len(result.results) == 1
        assert result.results[0].relevance_score == 0.9

        call_args = mock_client.post.call_args
        assert "/v1/rerank" in call_args[0][0]


@pytest.mark.asyncio
async def test_gateway_rerank_passes_top_n() -> None:
    """Verify top_n and max_tokens_per_doc are passed in the request body."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "rerank-456",
            "results": [{"index": 0, "relevance_score": 0.9}],
        }
        mock_client.post.return_value = mock_resp

        from any_llm.providers.gateway import GatewayProvider

        provider = GatewayProvider(api_key="test-key", api_base="http://localhost:8000")
        await provider._arerank("rerank-v3.5", "query", ["doc1"], top_n=5, max_tokens_per_doc=256)

        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        assert body["top_n"] == 5
        assert body["max_tokens_per_doc"] == 256


@pytest.mark.asyncio
async def test_gateway_rerank_strips_v1_suffix() -> None:
    """Verify that /v1 suffix is stripped from base URL before building the rerank URL."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "id": "rerank-789",
            "results": [],
        }
        mock_client.post.return_value = mock_resp

        from any_llm.providers.gateway import GatewayProvider

        provider = GatewayProvider(api_key="test-key", api_base="http://localhost:8000")
        await provider._arerank("model", "query", [])

        call_args = mock_client.post.call_args
        url = call_args[0][0]
        assert url == "http://localhost:8000/v1/rerank"
        assert "/v1/v1/" not in url


@pytest.mark.asyncio
async def test_rerank_api_function() -> None:
    """Test the public rerank() function dispatches correctly."""
    mock_provider = Mock()
    mock_rerank_response = RerankResponse(
        id="test-id",
        results=[RerankResult(index=0, relevance_score=0.9)],
    )
    mock_provider._rerank = Mock(return_value=mock_rerank_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        from any_llm.api import rerank

        result = rerank(
            "cohere:rerank-v3.5",
            query="test query",
            documents=["doc1", "doc2"],
            top_n=2,
            api_key="test_key",
        )

        assert result.id == "test-id"
        call_args = mock_provider._rerank.call_args
        assert call_args[0][0] == "rerank-v3.5"
        assert call_args[0][1] == "test query"
        assert call_args[0][2] == ["doc1", "doc2"]
        assert call_args[1]["top_n"] == 2


@pytest.mark.asyncio
async def test_arerank_api_function() -> None:
    """Test the public arerank() function dispatches correctly."""
    mock_provider = Mock()
    mock_rerank_response = RerankResponse(
        id="test-id",
        results=[RerankResult(index=0, relevance_score=0.9)],
    )
    mock_provider._arerank = AsyncMock(return_value=mock_rerank_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        from any_llm.api import arerank

        result = await arerank(
            "cohere:rerank-v3.5",
            query="test query",
            documents=["doc1", "doc2"],
            max_tokens_per_doc=512,
            api_key="test_key",
        )

        assert result.id == "test-id"
        call_args = mock_provider._arerank.call_args
        assert call_args[0][0] == "rerank-v3.5"
        assert call_args[1]["max_tokens_per_doc"] == 512


def test_rerank_response_model_validate() -> None:
    """Test that RerankResponse can be validated from a dict (gateway wire format)."""
    data = {
        "id": "rerank-wire",
        "results": [
            {"index": 0, "relevance_score": 0.95},
            {"index": 1, "relevance_score": 0.42},
        ],
        "usage": {"total_tokens": 75},
    }
    resp = RerankResponse.model_validate(data)
    assert resp.id == "rerank-wire"
    assert len(resp.results) == 2
    assert resp.usage is not None
    assert resp.usage.total_tokens == 75


def test_rerank_response_model_validate_minimal() -> None:
    """Test that minimal RerankResponse validates (no usage, no meta)."""
    data = {
        "id": "min",
        "results": [],
    }
    resp = RerankResponse.model_validate(data)
    assert resp.id == "min"
    assert resp.results == []
    assert resp.meta is None
    assert resp.usage is None


def test_openai_base_rerank_params_raises() -> None:
    from any_llm.providers.openai.base import BaseOpenAIProvider

    with pytest.raises(NotImplementedError, match="OpenAI does not support rerank"):
        BaseOpenAIProvider._convert_rerank_params("model", "query", ["doc"])


def test_openai_base_rerank_response_raises() -> None:
    from any_llm.providers.openai.base import BaseOpenAIProvider

    with pytest.raises(NotImplementedError, match="OpenAI does not support rerank"):
        BaseOpenAIProvider._convert_rerank_response({})


@pytest.mark.asyncio
async def test_cohere_arerank_calls_client() -> None:
    """Test that CohereProvider._arerank calls client.rerank with converted params."""
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.95

    mock_response = MagicMock()
    mock_response.id = "cohere-rerank-id"
    mock_response.results = [mock_result]
    mock_response.meta = None

    provider = CohereProvider.__new__(CohereProvider)
    provider.client = MagicMock()
    provider.client.rerank = AsyncMock(return_value=mock_response)

    result = await provider._arerank("rerank-v3.5", "my query", ["doc1", "doc2"], top_n=1)

    provider.client.rerank.assert_called_once()
    call_kwargs = provider.client.rerank.call_args[1]
    assert call_kwargs["model"] == "rerank-v3.5"
    assert call_kwargs["query"] == "my query"
    assert call_kwargs["documents"] == ["doc1", "doc2"]
    assert call_kwargs["top_n"] == 1

    assert isinstance(result, RerankResponse)
    assert result.id == "cohere-rerank-id"


def test_rerank_types_importable_from_types_package() -> None:
    from any_llm.types import RerankMeta, RerankResponse, RerankResult, RerankUsage

    assert RerankResponse is not None
    assert RerankResult is not None
    assert RerankMeta is not None
    assert RerankUsage is not None


def test_rerank_importable_from_top_level() -> None:
    from any_llm import arerank, rerank

    assert callable(rerank)
    assert callable(arerank)


def test_provider_metadata_rerank_field() -> None:
    from any_llm.types.provider import ProviderMetadata

    meta = ProviderMetadata(
        name="test",
        env_key="TEST_API_KEY",
        env_api_base=None,
        doc_url="https://example.com",
        streaming=True,
        reasoning=False,
        completion=True,
        embedding=False,
        responses=False,
        image=False,
        pdf=False,
        list_models=False,
        messages=False,
        batch_completion=False,
        rerank=True,
        class_name="TestProvider",
    )
    assert meta.rerank is True


def test_anthropic_rerank_params_raises() -> None:
    from any_llm.providers.anthropic.base import BaseAnthropicProvider

    with pytest.raises(NotImplementedError, match="Anthropic does not support rerank"):
        BaseAnthropicProvider._convert_rerank_params("model", "query", ["doc"])


def test_anthropic_rerank_response_raises() -> None:
    from any_llm.providers.anthropic.base import BaseAnthropicProvider

    with pytest.raises(NotImplementedError, match="Anthropic does not support rerank"):
        BaseAnthropicProvider._convert_rerank_response({})


def test_mistral_rerank_params_raises() -> None:
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with pytest.raises(NotImplementedError, match="Mistral does not support rerank"):
        MistralProvider._convert_rerank_params("model", "query", ["doc"])


def test_mistral_rerank_response_raises() -> None:
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    with pytest.raises(NotImplementedError, match="Mistral does not support rerank"):
        MistralProvider._convert_rerank_response({})


def test_mistral_does_not_support_rerank() -> None:
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider

    assert MistralProvider.SUPPORTS_RERANK is False


def test_azure_rerank_params_raises() -> None:
    from any_llm.providers.azure.azure import AzureProvider

    with pytest.raises(NotImplementedError, match="Azure does not support rerank"):
        AzureProvider._convert_rerank_params("model", "query", ["doc"])


def test_azure_rerank_response_raises() -> None:
    from any_llm.providers.azure.azure import AzureProvider

    with pytest.raises(NotImplementedError, match="Azure does not support rerank"):
        AzureProvider._convert_rerank_response({})


def test_azure_does_not_support_rerank() -> None:
    from any_llm.providers.azure.azure import AzureProvider

    assert AzureProvider.SUPPORTS_RERANK is False


def test_bedrock_rerank_params_raises() -> None:
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    with pytest.raises(NotImplementedError, match="Bedrock does not support rerank"):
        BedrockProvider._convert_rerank_params("model", "query", ["doc"])


def test_bedrock_rerank_response_raises() -> None:
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    with pytest.raises(NotImplementedError, match="Bedrock does not support rerank"):
        BedrockProvider._convert_rerank_response({})


def test_bedrock_does_not_support_rerank() -> None:
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    assert BedrockProvider.SUPPORTS_RERANK is False


def test_cerebras_rerank_params_raises() -> None:
    from any_llm.providers.cerebras.cerebras import CerebrasProvider

    with pytest.raises(NotImplementedError, match="Cerebras does not support rerank"):
        CerebrasProvider._convert_rerank_params("model", "query", ["doc"])


def test_cerebras_rerank_response_raises() -> None:
    from any_llm.providers.cerebras.cerebras import CerebrasProvider

    with pytest.raises(NotImplementedError, match="Cerebras does not support rerank"):
        CerebrasProvider._convert_rerank_response({})


def test_cerebras_does_not_support_rerank() -> None:
    from any_llm.providers.cerebras.cerebras import CerebrasProvider

    assert CerebrasProvider.SUPPORTS_RERANK is False


def test_groq_rerank_params_raises() -> None:
    from any_llm.providers.groq.groq import GroqProvider

    with pytest.raises(NotImplementedError, match="Groq does not support rerank"):
        GroqProvider._convert_rerank_params("model", "query", ["doc"])


def test_groq_rerank_response_raises() -> None:
    from any_llm.providers.groq.groq import GroqProvider

    with pytest.raises(NotImplementedError, match="Groq does not support rerank"):
        GroqProvider._convert_rerank_response({})


def test_groq_does_not_support_rerank() -> None:
    from any_llm.providers.groq.groq import GroqProvider

    assert GroqProvider.SUPPORTS_RERANK is False


def test_huggingface_rerank_params_raises() -> None:
    from any_llm.providers.huggingface.huggingface import HuggingfaceProvider

    with pytest.raises(NotImplementedError, match="HuggingFace does not support rerank"):
        HuggingfaceProvider._convert_rerank_params("model", "query", ["doc"])


def test_huggingface_rerank_response_raises() -> None:
    from any_llm.providers.huggingface.huggingface import HuggingfaceProvider

    with pytest.raises(NotImplementedError, match="HuggingFace does not support rerank"):
        HuggingfaceProvider._convert_rerank_response({})


def test_huggingface_does_not_support_rerank() -> None:
    from any_llm.providers.huggingface.huggingface import HuggingfaceProvider

    assert HuggingfaceProvider.SUPPORTS_RERANK is False


def test_ollama_rerank_params_raises() -> None:
    from any_llm.providers.ollama.ollama import OllamaProvider

    with pytest.raises(NotImplementedError, match="Ollama does not support rerank"):
        OllamaProvider._convert_rerank_params("model", "query", ["doc"])


def test_ollama_rerank_response_raises() -> None:
    from any_llm.providers.ollama.ollama import OllamaProvider

    with pytest.raises(NotImplementedError, match="Ollama does not support rerank"):
        OllamaProvider._convert_rerank_response({})


def test_ollama_does_not_support_rerank() -> None:
    from any_llm.providers.ollama.ollama import OllamaProvider

    assert OllamaProvider.SUPPORTS_RERANK is False


def test_sagemaker_rerank_params_raises() -> None:
    from any_llm.providers.sagemaker.sagemaker import SagemakerProvider

    with pytest.raises(NotImplementedError, match="Sagemaker does not support rerank"):
        SagemakerProvider._convert_rerank_params("model", "query", ["doc"])


def test_sagemaker_rerank_response_raises() -> None:
    from any_llm.providers.sagemaker.sagemaker import SagemakerProvider

    with pytest.raises(NotImplementedError, match="Sagemaker does not support rerank"):
        SagemakerProvider._convert_rerank_response({})


def test_sagemaker_does_not_support_rerank() -> None:
    from any_llm.providers.sagemaker.sagemaker import SagemakerProvider

    assert SagemakerProvider.SUPPORTS_RERANK is False


def test_together_rerank_params_raises() -> None:
    from any_llm.providers.together.together import TogetherProvider

    with pytest.raises(NotImplementedError, match="Together does not support rerank"):
        TogetherProvider._convert_rerank_params("model", "query", ["doc"])


def test_together_rerank_response_raises() -> None:
    from any_llm.providers.together.together import TogetherProvider

    with pytest.raises(NotImplementedError, match="Together does not support rerank"):
        TogetherProvider._convert_rerank_response({})


def test_together_does_not_support_rerank() -> None:
    from any_llm.providers.together.together import TogetherProvider

    assert TogetherProvider.SUPPORTS_RERANK is False


def test_voyage_rerank_params_raises() -> None:
    from any_llm.providers.voyage.voyage import VoyageProvider

    with pytest.raises(NotImplementedError, match="Voyage does not support rerank"):
        VoyageProvider._convert_rerank_params("model", "query", ["doc"])


def test_voyage_rerank_response_raises() -> None:
    from any_llm.providers.voyage.voyage import VoyageProvider

    with pytest.raises(NotImplementedError, match="Voyage does not support rerank"):
        VoyageProvider._convert_rerank_response({})


def test_voyage_does_not_support_rerank() -> None:
    from any_llm.providers.voyage.voyage import VoyageProvider

    assert VoyageProvider.SUPPORTS_RERANK is False


def test_watsonx_rerank_params_raises() -> None:
    from any_llm.providers.watsonx.watsonx import WatsonxProvider

    with pytest.raises(NotImplementedError, match="Watsonx does not support rerank"):
        WatsonxProvider._convert_rerank_params("model", "query", ["doc"])


def test_watsonx_rerank_response_raises() -> None:
    from any_llm.providers.watsonx.watsonx import WatsonxProvider

    with pytest.raises(NotImplementedError, match="Watsonx does not support rerank"):
        WatsonxProvider._convert_rerank_response({})


def test_watsonx_does_not_support_rerank() -> None:
    from any_llm.providers.watsonx.watsonx import WatsonxProvider

    assert WatsonxProvider.SUPPORTS_RERANK is False


def test_xai_rerank_params_raises() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with pytest.raises(NotImplementedError, match="XAI does not support rerank"):
        XaiProvider._convert_rerank_params("model", "query", ["doc"])


def test_xai_rerank_response_raises() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    with pytest.raises(NotImplementedError, match="XAI does not support rerank"):
        XaiProvider._convert_rerank_response({})


def test_xai_does_not_support_rerank() -> None:
    from any_llm.providers.xai.xai import XaiProvider

    assert XaiProvider.SUPPORTS_RERANK is False


def test_gateway_convert_rerank_params_raises() -> None:
    from any_llm.providers.gateway import GatewayProvider

    with pytest.raises(NotImplementedError, match="Gateway rerank uses direct HTTP"):
        GatewayProvider._convert_rerank_params("model", "query", ["doc"])


def test_gateway_convert_rerank_response_raises() -> None:
    from any_llm.providers.gateway import GatewayProvider

    with pytest.raises(NotImplementedError, match="Gateway rerank uses direct HTTP"):
        GatewayProvider._convert_rerank_response({})


def test_rerank_api_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_rerank_response = RerankResponse(
        id="explicit-prov",
        results=[RerankResult(index=0, relevance_score=0.8)],
    )
    mock_provider._rerank = Mock(return_value=mock_rerank_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        from any_llm.api import rerank

        result = rerank(
            "rerank-v3.5",
            query="test query",
            documents=["doc1"],
            provider="cohere",
            max_tokens_per_doc=256,
            api_key="test_key",
        )

        assert result.id == "explicit-prov"
        call_args = mock_provider._rerank.call_args
        assert call_args[1]["max_tokens_per_doc"] == 256


@pytest.mark.asyncio
async def test_arerank_api_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_rerank_response = RerankResponse(
        id="async-explicit",
        results=[RerankResult(index=0, relevance_score=0.7)],
    )
    mock_provider._arerank = AsyncMock(return_value=mock_rerank_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        from any_llm.api import arerank

        result = await arerank(
            "rerank-v3.5",
            query="test",
            documents=["d1"],
            provider="cohere",
            top_n=3,
            api_key="key",
        )

        assert result.id == "async-explicit"
        call_args = mock_provider._arerank.call_args
        assert call_args[1]["top_n"] == 3


def test_rerank_api_with_client_args() -> None:
    mock_provider = Mock()
    mock_rerank_response = RerankResponse(id="ca", results=[])
    mock_provider._rerank = Mock(return_value=mock_rerank_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        from any_llm.api import rerank

        rerank(
            "cohere:rerank-v3.5",
            query="q",
            documents=[],
            client_args={"timeout": 30},
            api_key="k",
        )

        create_call = mock_create.call_args
        assert create_call[1]["timeout"] == 30


@pytest.mark.asyncio
async def test_gateway_rerank_platform_mode_http_error() -> None:
    """Gateway rerank wraps HTTPStatusError as openai.APIStatusError in platform mode."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        mock_response.stream = MagicMock()
        mock_response.is_closed = True
        http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)
        mock_client.post.return_value = MagicMock()
        mock_client.post.return_value.raise_for_status.side_effect = http_error

        from any_llm.providers.gateway import GatewayProvider

        provider = GatewayProvider(api_key="test-key", api_base="http://localhost:8000")
        provider.platform_mode = True

        import openai

        with pytest.raises(openai.APIStatusError):
            await provider._arerank("model", "query", ["doc"])


@pytest.mark.asyncio
async def test_gateway_rerank_platform_mode_generic_error() -> None:
    """Gateway rerank calls _handle_platform_error for generic exceptions in platform mode."""
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_client.post.side_effect = ConnectionError("connection failed")

        from any_llm.providers.gateway import GatewayProvider

        provider = GatewayProvider(api_key="test-key", api_base="http://localhost:8000")
        provider.platform_mode = True

        with patch.object(provider, "_handle_platform_error") as mock_handle:
            with pytest.raises(ConnectionError):
                await provider._arerank("model", "query", ["doc"])

            mock_handle.assert_called_once()


@pytest.mark.asyncio
async def test_gateway_rerank_non_platform_http_error_reraises() -> None:
    """Gateway rerank re-raises HTTPStatusError when not in platform mode."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        http_error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response)
        mock_client.post.return_value = MagicMock()
        mock_client.post.return_value.raise_for_status.side_effect = http_error

        from any_llm.providers.gateway import GatewayProvider

        provider = GatewayProvider(api_key="test-key", api_base="http://localhost:8000")
        provider.platform_mode = False

        with pytest.raises(httpx.HTTPStatusError):
            await provider._arerank("model", "query", ["doc"])


def test_cohere_convert_rerank_params_ignores_none_max_tokens() -> None:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere import CohereProvider

    params = CohereProvider._convert_rerank_params(
        model="rerank-v3.5",
        query="q",
        documents=["d"],
        max_tokens_per_doc=None,
    )
    assert "max_tokens_per_doc" not in params


def test_convert_cohere_rerank_response_meta_no_search_units() -> None:
    """Cohere response with billed_units that has no search_units attribute."""
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.utils import _convert_cohere_rerank_response

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.7

    mock_meta = MagicMock()
    mock_meta.billed_units = MagicMock(spec=[])
    del mock_meta.billed_units.search_units
    mock_meta.tokens = MagicMock()
    mock_meta.tokens.input_tokens = 50

    mock_response = MagicMock()
    mock_response.id = "no-su"
    mock_response.results = [mock_result]
    mock_response.meta = mock_meta

    result = _convert_cohere_rerank_response(mock_response)
    assert result.meta is not None
    assert result.meta.tokens is not None
    assert result.meta.tokens["input_tokens"] == 50
    assert result.usage is not None
    assert result.usage.total_tokens == 50


def test_convert_cohere_rerank_response_meta_no_input_tokens() -> None:
    """Cohere response with tokens that has no input_tokens attribute."""
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.utils import _convert_cohere_rerank_response

    mock_result = MagicMock()
    mock_result.index = 0
    mock_result.relevance_score = 0.6

    mock_meta = MagicMock()
    mock_meta.billed_units = MagicMock()
    mock_meta.billed_units.search_units = 2.0
    mock_meta.tokens = MagicMock(spec=[])
    del mock_meta.tokens.input_tokens

    mock_response = MagicMock()
    mock_response.id = "no-it"
    mock_response.results = [mock_result]
    mock_response.meta = mock_meta

    result = _convert_cohere_rerank_response(mock_response)
    assert result.meta is not None
    assert result.meta.billed_units is not None
    assert result.meta.billed_units["search_units"] == 2.0
    assert result.usage is None


def test_gemini_rerank_params_raises() -> None:
    try:
        from any_llm.providers.gemini.base import GoogleProvider
    except ImportError:
        pytest.skip("google-genai not installed")

    with pytest.raises(NotImplementedError, match="Google does not support rerank"):
        GoogleProvider._convert_rerank_params("model", "query", ["doc"])


def test_gemini_rerank_response_raises() -> None:
    try:
        from any_llm.providers.gemini.base import GoogleProvider
    except ImportError:
        pytest.skip("google-genai not installed")

    with pytest.raises(NotImplementedError, match="Google does not support rerank"):
        GoogleProvider._convert_rerank_response({})


def test_gemini_does_not_support_rerank() -> None:
    try:
        from any_llm.providers.gemini.base import GoogleProvider
    except ImportError:
        pytest.skip("google-genai not installed")

    assert GoogleProvider.SUPPORTS_RERANK is False


def test_platform_rerank_params_raises() -> None:
    try:
        from any_llm.providers.platform.platform import PlatformProvider
    except ImportError:
        pytest.skip("any-llm-platform-client not installed")

    with pytest.raises(NotImplementedError, match="Platform does not support rerank"):
        PlatformProvider._convert_rerank_params("model", "query", ["doc"])


def test_platform_rerank_response_raises() -> None:
    try:
        from any_llm.providers.platform.platform import PlatformProvider
    except ImportError:
        pytest.skip("any-llm-platform-client not installed")

    with pytest.raises(NotImplementedError, match="Platform does not support rerank"):
        PlatformProvider._convert_rerank_response({})
