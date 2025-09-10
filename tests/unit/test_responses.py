from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import aresponses
from any_llm.constants import ProviderName


@pytest.mark.asyncio
async def test_responses_invalid_model_format_no_slash() -> None:
    """Test responses raises ValueError for model without separator."""
    with pytest.raises(
        ValueError, match=r"Invalid model format. Expected 'provider:model' or 'provider/model', got 'gpt-5-nano'"
    ):
        await aresponses("gpt-5-nano", input_data=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_responses_invalid_model_format_empty_provider() -> None:
    """Test responses raises ValueError for model with empty provider."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await aresponses("/model", input_data=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_responses_invalid_model_format_empty_model() -> None:
    """Test responses raises ValueError for model with empty model name."""
    with pytest.raises(ValueError, match="Invalid model format"):
        await aresponses("provider/", input_data=[{"role": "user", "content": "Hello"}])


@pytest.mark.asyncio
async def test_responses_invalid_model_format_multiple_slashes() -> None:
    """Test responses handles multiple slashes correctly (should work - takes first split)."""
    mock_provider = Mock()
    mock_provider.aresponses = AsyncMock()

    with patch("any_llm.provider.Provider.create") as mock_create:
        mock_create.return_value = mock_provider

        await aresponses("openai/model/extra", input_data=[{"role": "user", "content": "Hello"}])

        mock_provider.aresponses.assert_called_once_with("model/extra", [{"role": "user", "content": "Hello"}])
