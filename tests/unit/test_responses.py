import pytest

from any_llm.api import aresponses


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
