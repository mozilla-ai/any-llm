from datetime import datetime
from unittest.mock import Mock

from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.providers.together.utils import _create_openai_chunk_from_together_chunk
from any_llm.types.completion import ChatCompletionChunk


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
