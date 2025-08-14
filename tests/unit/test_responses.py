from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import aresponses, responses
from any_llm.provider import ProviderName

INPUT_DATA = [{"role": "user", "content": "Hello"}]
INPUT_KWARGS = {
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "existing_tool",
                "description": "An existing tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ],
    "tool_choice": "auto",
    "max_output_tokens": 512,
    "temperature": 0.5,
    "top_p": 0.5,
    "max_tool_calls": 3,
    "parallel_tool_calls": True,
    "reasoning": {"effort": "medium"},
    "text": {
        "verbosity": "low",
    },
    "api_key": "asdf",
    "api_base": "http://localhost:3000",
    "instructions": "Talk like a pirate.",
    "user": "foo",
}

# These disappear in responses call
OMIT_MOCK_FIELDS = ["api_key", "api_base"]


def test_responses_invalid_model_format_no_slash() -> None:
    """Test responses raises ValueError for model without slash."""
    with pytest.raises(ValueError, match="Invalid model format. Expected 'provider/model', got 'gpt-5-nano'"):
        responses("gpt-5-nano", INPUT_DATA, **INPUT_KWARGS)


def test_responses_invalid_model_format_empty_provider() -> None:
    """Test responses raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        responses("/model", INPUT_DATA, **INPUT_KWARGS)


def test_responses_invalid_model_format_empty_model() -> None:
    """Test responses raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        responses("provider/", INPUT_DATA, **INPUT_KWARGS)


def test_responses_invalid_model_format_multiple_slashes() -> None:
    """Test responses handles multiple slashes correctly (should work - takes first split)."""
    mock_provider = Mock()
    mock_provider.responses.return_value = Mock()

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI  # Using a valid provider
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "model/extra")
        mock_factory.create_provider.return_value = mock_provider

        responses("provider/model/extra", input_data=INPUT_DATA, **INPUT_KWARGS)

        mock_provider.responses.assert_called_once_with(
            "model/extra", INPUT_DATA, **{i: j for i, j in INPUT_KWARGS.items() if i not in OMIT_MOCK_FIELDS}
        )


@pytest.mark.asyncio
async def test_aresponses_invalid_model_format_multiple_slashes() -> None:
    """Test responses handles multiple slashes correctly (should work - takes first split)."""
    mock_provider = AsyncMock()
    mock_provider.responses.return_value = AsyncMock()

    with patch("any_llm.api.ProviderFactory") as mock_factory:
        mock_factory.get_supported_providers.return_value = ["provider"]
        mock_factory.get_provider_enum.return_value = ProviderName.OPENAI  # Using a valid provider
        mock_factory.split_model_provider.return_value = (ProviderName.OPENAI, "model/extra")
        mock_factory.create_provider.return_value = mock_provider

        await aresponses("provider/model/extra", input_data=INPUT_DATA, **INPUT_KWARGS)

        mock_provider.aresponses.assert_called_once_with(
            "model/extra", INPUT_DATA, **{i: j for i, j in INPUT_KWARGS.items() if i not in OMIT_MOCK_FIELDS}
        )
