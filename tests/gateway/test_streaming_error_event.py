"""Tests for SSE error event emission on streaming failures."""

import json
from typing import Any

from fastapi.testclient import TestClient


def test_streaming_error_emits_sse_error_event(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a streaming request to an invalid model emits an SSE error event."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:totally-invalid-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
            "user": test_user["user_id"],
            "stream": True,
        },
        headers=api_key_header,
    )

    # The response should be 200 (streaming started) but contain an error event
    assert response.status_code == 200

    events: list[str] = []
    found_error_event = False
    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            events.append(data_str)
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if "error" in chunk:
                    found_error_event = True
                    assert "message" in chunk["error"]
                    assert "type" in chunk["error"]
            except json.JSONDecodeError:
                continue

    assert found_error_event, "Should have received an SSE error event for invalid model"

    # [DONE] must be the last event and must come after the error event
    assert events[-1] == "[DONE]", "Stream should end with [DONE] after error event"

    # No [DONE] should appear before the error event
    done_indices = [i for i, e in enumerate(events) if e == "[DONE]"]
    error_indices = [i for i, e in enumerate(events) if e != "[DONE]" and "error" in e]
    assert error_indices, "Should have found an error event"
    assert done_indices[-1] > error_indices[0], "[DONE] must come after the error event"
