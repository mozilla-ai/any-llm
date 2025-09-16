import warnings
from unittest.mock import Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import completion
from any_llm.constants import LLMProvider
from any_llm.exceptions import UnsupportedProviderError


def test_colon_syntax_valid() -> None:
    """Test that the new colon syntax works correctly."""
    result = AnyLLM.split_model_provider("openai:gpt-4")
    assert result == (LLMProvider.OPENAI, "gpt-4")


def test_slash_syntax_shows_deprecation_warning() -> None:
    """Test that slash syntax shows deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = AnyLLM.split_model_provider("openai/gpt-4")
        assert result == (LLMProvider.OPENAI, "gpt-4")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message)
        assert "version 1.0" in str(w[0].message)
        assert "openai/gpt-4" in str(w[0].message)


def test_completion_with_colon_syntax() -> None:
    """Test that completion function works with colon syntax."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        completion("openai:gpt-4", messages=[{"role": "user", "content": "Hello"}])
        mock_provider.completion.assert_called_once()


def test_completion_with_separate_parameters() -> None:
    """Test that completion function works with separate provider and model parameters (recommended approach)."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        completion(model="gpt-4", provider="openai", messages=[{"role": "user", "content": "Hello"}])
        mock_provider.completion.assert_called_once()


def test_completion_with_slash_syntax_shows_warning() -> None:
    """Test that completion function shows deprecation warning with slash syntax."""
    mock_provider = Mock()
    mock_provider.completion.return_value = Mock()

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
            mock_create.return_value = mock_provider

            completion("openai/gpt-4", messages=[{"role": "user", "content": "Hello"}])
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            mock_provider.completion.assert_called_once()


def test_provider_name_from_string_with_enum() -> None:
    """Test that from_string works with ProviderName enum."""
    result = LLMProvider.from_string(LLMProvider.OPENAI)
    assert result == LLMProvider.OPENAI


def test_provider_name_from_string_with_valid_string() -> None:
    """Test that from_string works with valid string."""
    result = LLMProvider.from_string("openai")
    assert result == LLMProvider.OPENAI


def test_provider_name_from_string_with_uppercase_string() -> None:
    """Test that from_string works with uppercase string."""
    result = LLMProvider.from_string("OPENAI")
    assert result == LLMProvider.OPENAI


def test_provider_name_from_string_with_invalid_string() -> None:
    """Test that from_string raises UnsupportedProviderError for invalid string."""
    with pytest.raises(UnsupportedProviderError) as exc_info:
        LLMProvider.from_string("invalid_provider")

    assert "invalid_provider" in str(exc_info.value)
