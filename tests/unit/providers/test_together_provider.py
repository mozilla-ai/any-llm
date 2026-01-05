from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.providers.together.utils import _create_openai_chunk_from_together_chunk
from any_llm.types.completion import ChatCompletionChunk, CompletionParams


def test_create_openai_chunk_handles_empty_choices() -> None:
    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = None
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert isinstance(result, ChatCompletionChunk)
    assert result.choices == []
    assert result.id == "test-id"
    assert result.model == "test-model"

    together_chunk.choices = []
    result = _create_openai_chunk_from_together_chunk(together_chunk)
    assert result.choices == []


def test_create_openai_chunk_handles_missing_delta() -> None:
    """Test that the function handles choices with None delta gracefully."""
    choice_mock = Mock()
    choice_mock.delta = None
    choice_mock.index = 0
    choice_mock.finish_reason = "stop"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.content is None
    assert result.choices[0].delta.role is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Together API calls."""
    from any_llm.providers.together.together import TogetherProvider

    with (
        patch("together.AsyncTogether") as mock_together,
        patch.object(TogetherProvider, "_convert_completion_response", return_value=Mock()),
    ):
        mock_client = Mock()
        mock_together.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=Mock())

        provider = TogetherProvider(api_key="test-api-key")
        await provider._acompletion(
            CompletionParams(
                model_id="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            ),
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs
