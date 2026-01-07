from unittest.mock import AsyncMock, Mock, patch

import pytest
from ollama import ChatResponse as OllamaChatResponse
from ollama import Message as OllamaMessage

from any_llm.providers.ollama.ollama import OllamaProvider
from any_llm.providers.ollama.utils import _create_chat_completion_from_ollama_response
from any_llm.types.completion import CompletionParams


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


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Ollama API calls."""
    with patch.object(OllamaProvider, "_init_client"):
        provider = OllamaProvider(api_key=None)
        provider.client = Mock()
        provider.client.chat = AsyncMock(return_value=Mock())

        with patch.object(OllamaProvider, "_convert_completion_response", return_value=Mock()):
            await provider._acompletion(
                CompletionParams(
                    model_id="llama3.1",
                    messages=[{"role": "user", "content": "Hello"}],
                    reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
                ),
            )

            call_kwargs = provider.client.chat.call_args[1]
            assert "reasoning_effort" not in call_kwargs.get("options", {})
