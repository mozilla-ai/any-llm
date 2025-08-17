from collections.abc import Sequence
from unittest.mock import MagicMock, patch

import pytest

from any_llm.provider import ApiConfig
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.model import ModelMetadata


class DummyConfig(ApiConfig):
    api_key: str | None = "test-key"
    api_base: str | None = "https://api.openai.com/v1"


class TestBaseOpenAIProvider:
    @patch("any_llm.providers.openai.base.OpenAI")
    def test_models_returns_model_list(self, mock_openai: MagicMock) -> None:
        # Arrange
        class FakeProvider(BaseOpenAIProvider):
            SUPPORTS_LIST_MODELS = True
            PROVIDER_NAME = "FakeProvider"

        provider = FakeProvider(config=DummyConfig())
        mock_openai_model_data = []
        for d in [
            {"id": "model-id-0", "object": "model", "created": 1686935002},
            {"id": "model-id-1", "object": "model", "created": 1686935002},
        ]:
            m = MagicMock()
            m.model_dump.return_value = d
            mock_openai_model_data.append(m)
        mock_openai_list_response = MagicMock()
        mock_openai_list_response.data = mock_openai_model_data
        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_openai_list_response
        mock_openai.return_value = mock_client

        # Act
        result = provider.models()

        # Assert
        assert isinstance(result, Sequence)
        assert len(result) == 2
        assert all(isinstance(m, ModelMetadata) for m in result)
        assert result[0].id == "model-id-0"
        assert result[1].id == "model-id-1"

    def test_models_raises_not_implemented_if_not_supported(self) -> None:
        class NoListModelsProvider(BaseOpenAIProvider):
            SUPPORTS_LIST_MODELS = False
            PROVIDER_NAME = "NoListModelsProvider"

        provider = NoListModelsProvider(config=DummyConfig())
        with pytest.raises(NotImplementedError):
            provider.models()
