from unittest.mock import Mock

from any_llm.providers.fireworks.utils import extract_reasoning_from_response
from any_llm.types.completion import Reasoning
from any_llm.types.responses import Response


def test_extract_reasoning_from_response_with_think_tags() -> None:
    """Test that <think> content is correctly extracted into reasoning field."""
    # Create a mock Response with <think> tags in content
    mock_content = Mock()
    mock_content.text = "<think>This is my reasoning process</think>This is the actual response"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is not None
    assert isinstance(result.reasoning, Reasoning)
    assert result.reasoning.content == "This is my reasoning process"
    assert mock_content.text == "This is the actual response"


def test_extract_reasoning_from_response_without_think_tags() -> None:
    """Test that responses without <think> tags are returned unchanged."""
    mock_content = Mock()
    mock_content.text = "This is just a regular response"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is None
    assert mock_content.text == "This is just a regular response"


def test_extract_reasoning_from_response_empty_reasoning() -> None:
    """Test that empty reasoning content is handled correctly."""
    mock_content = Mock()
    mock_content.text = "<think></think>This is the actual response"

    mock_output_item = Mock()
    mock_output_item.content = [mock_content]

    mock_response = Mock(spec=Response)
    mock_response.output = [mock_output_item]
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is None
    assert mock_content.text == "This is the actual response"


def test_extract_reasoning_from_response_empty_output() -> None:
    """Test that responses with empty output are handled gracefully."""
    mock_response = Mock(spec=Response)
    mock_response.output = []
    mock_response.reasoning = None

    result = extract_reasoning_from_response(mock_response)

    assert result.reasoning is None
    assert result == mock_response
