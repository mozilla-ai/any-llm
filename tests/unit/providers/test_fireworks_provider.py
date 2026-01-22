from typing import Any

from openresponses_types import ResponseResource

from any_llm.providers.fireworks.utils import extract_reasoning_from_response


def _create_test_response(text: str) -> ResponseResource:
    """Create a minimal valid ResponseResource for testing."""
    data: dict[str, Any] = {
        "id": "test-id",
        "object": "response",
        "created_at": 1234567890,
        "completed_at": None,
        "status": "completed",
        "incomplete_details": None,
        "model": "test-model",
        "previous_response_id": None,
        "instructions": None,
        "output": [
            {
                "type": "message",
                "id": "msg-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
            }
        ],
        "error": None,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "auto",
        "parallel_tool_calls": True,
        "text": {"format": {"type": "text"}},
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "temperature": 1.0,
        "reasoning": None,
        "usage": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": True,
        "background": False,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }
    return ResponseResource.model_validate(data)


def test_extract_reasoning_from_response_with_think_tags() -> None:
    """Test that <think> content is correctly extracted and text is cleaned."""
    response = _create_test_response("<think>This is my reasoning process</think>This is the actual response")

    result = extract_reasoning_from_response(response)

    # The text should be cleaned (think tags removed)
    assert result.output[0].content[0].text == "This is the actual response"


def test_extract_reasoning_from_response_without_think_tags() -> None:
    """Test that responses without <think> tags are returned unchanged."""
    response = _create_test_response("This is just a regular response")

    result = extract_reasoning_from_response(response)

    assert result.output[0].content[0].text == "This is just a regular response"
    assert result is response  # Same object returned when no changes needed


def test_extract_reasoning_from_response_empty_reasoning() -> None:
    """Test that empty reasoning content is handled correctly."""
    response = _create_test_response("<think></think>This is the actual response")

    result = extract_reasoning_from_response(response)

    assert result.output[0].content[0].text == "This is the actual response"


def test_extract_reasoning_from_response_empty_output() -> None:
    """Test that responses with empty output are handled gracefully."""
    data: dict[str, Any] = {
        "id": "test-id",
        "object": "response",
        "created_at": 1234567890,
        "completed_at": None,
        "status": "completed",
        "incomplete_details": None,
        "model": "test-model",
        "previous_response_id": None,
        "instructions": None,
        "output": [],
        "error": None,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "auto",
        "parallel_tool_calls": True,
        "text": {"format": {"type": "text"}},
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "top_logprobs": 0,
        "temperature": 1.0,
        "reasoning": None,
        "usage": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": True,
        "background": False,
        "service_tier": "default",
        "metadata": {},
        "safety_identifier": None,
        "prompt_cache_key": None,
    }
    response = ResponseResource.model_validate(data)

    result = extract_reasoning_from_response(response)

    assert result is response
