from unittest.mock import Mock

import pytest
from ollama import ChatResponse as OllamaChatResponse
from ollama import Message as OllamaMessage

from any_llm.providers.ollama.utils import _create_chat_completion_from_ollama_response


@pytest.mark.asyncio
async def test_create_chat_completion_extracts_think_content() -> None:
    """Test that <think> content is correctly extracted into reasoning field."""
    # Create a mock Ollama response with <think> tags in content
    mock_message = Mock(spec=OllamaMessage)
    mock_message.content = "<think>This is my reasoning process</think>This is the actual response"
    mock_message.thinking = None
    mock_message.tool_calls = None
    mock_message.role = "assistant"

    mock_response = Mock(spec=OllamaChatResponse)
    mock_response.message = mock_message
    mock_response.created_at = "2024-01-01T12:00:00.000000Z"
    mock_response.prompt_eval_count = 10
    mock_response.eval_count = 20
    mock_response.model = "llama3.1"
    mock_response.done_reason = "stop"

    result = _create_chat_completion_from_ollama_response(mock_response)

    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "This is my reasoning process"

    assert result.choices[0].message.content == "This is the actual response"
