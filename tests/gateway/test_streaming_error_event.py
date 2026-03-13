"""Tests for error handling on streaming requests."""

from typing import Any

from fastapi.testclient import TestClient


def test_streaming_creation_error_returns_http_error(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a streaming request to an invalid model returns an HTTP error.

    When the stream cannot be created (e.g., invalid model, missing API key),
    the gateway returns a proper HTTP error response rather than starting a
    stream and emitting an SSE error event.
    """
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

    assert response.status_code == 500
    assert "provider" in response.json()["detail"].lower()
