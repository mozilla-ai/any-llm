import pytest

from any_llm.api import aresponses
from any_llm.types.responses import ResponsesParams


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


def test_omitted_parallel_tool_calls_stays_none() -> None:
    """Omitted parallel_tool_calls should not appear in provider request."""
    params = ResponsesParams(model="test", input="hello")
    assert params.parallel_tool_calls is None
    assert "parallel_tool_calls" not in params.model_dump(exclude_none=True)


@pytest.mark.parametrize("value", [True, False])
def test_explicit_parallel_tool_calls_preserved(value: bool) -> None:
    """Explicit True/False should be kept as-is."""
    params = ResponsesParams(model="test", input="hello", parallel_tool_calls=value)
    assert params.parallel_tool_calls is value
