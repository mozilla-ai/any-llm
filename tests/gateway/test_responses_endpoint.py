"""Tests for the /v1/responses gateway endpoint."""

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from any_llm.gateway.core.config import API_KEY_HEADER


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int, total_tokens: int | None = None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens if total_tokens is not None else input_tokens + output_tokens


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], usage: _FakeUsage | None = None) -> None:
        self._payload = payload
        self.usage = usage

    def model_dump(self, *, exclude_none: bool = False) -> dict[str, Any]:
        return self._payload


class _FakeStreamEvent:
    def __init__(
        self,
        event_type: str,
        payload: dict[str, Any],
        response: _FakeResponse | None = None,
    ) -> None:
        self.type = event_type
        self._payload = payload
        self.response = response

    def model_dump_json(self, *, exclude_none: bool = False) -> str:
        return json.dumps(self._payload)


def test_responses_endpoint_basic_completion(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test basic non-streaming response creation."""
    mock_response = _FakeResponse(
        payload={
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4.1-mini",
            "output": [],
        },
        usage=_FakeUsage(input_tokens=10, output_tokens=5),
    )

    with patch("any_llm.gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/responses",
            json={
                "model": "openai:gpt-4.1-mini",
                "input": "Hello",
                "user": test_user["user_id"],
            },
            headers=master_key_header,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "resp_123"
    assert data["object"] == "response"
    assert data["model"] == "gpt-4.1-mini"


def test_responses_endpoint_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that master key auth requires user in the request body."""
    response = client.post(
        "/v1/responses",
        json={
            "model": "openai:gpt-4.1-mini",
            "input": "Hello",
        },
        headers=master_key_header,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "When using master key, 'user' field is required in request body"


def test_responses_endpoint_rejects_unsupported_provider(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that providers without responses support are rejected."""
    response = client.post(
        "/v1/responses",
        json={
            "model": "anthropic:claude-3-5-sonnet",
            "input": "Hello",
            "user": test_user["user_id"],
        },
        headers=master_key_header,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provider 'anthropic' does not support the Responses API"


def test_responses_endpoint_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test streaming responses use the native Responses SSE format."""
    completed_response = _FakeResponse(
        payload={
            "id": "resp_123",
            "object": "response",
            "model": "gpt-4.1-mini",
            "output": [],
        },
        usage=_FakeUsage(input_tokens=12, output_tokens=4),
    )

    async def mock_stream() -> AsyncIterator[_FakeStreamEvent]:
        yield _FakeStreamEvent(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "delta": "Hel",
                "item_id": "item_123",
                "output_index": 0,
                "content_index": 0,
            },
        )
        yield _FakeStreamEvent(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "object": "response",
                },
            },
            response=completed_response,
        )

    with patch("any_llm.gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=mock_stream()):
        response = client.post(
            "/v1/responses",
            json={
                "model": "openai:gpt-4.1-mini",
                "input": "Hello",
                "user": test_user["user_id"],
                "stream": True,
            },
            headers=master_key_header,
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    raw_lines = [line for line in response.iter_lines() if line]
    assert raw_lines[0] == "event: response.output_text.delta"
    assert raw_lines[1].startswith("data: ")
    assert json.loads(raw_lines[1][6:])["delta"] == "Hel"
    assert "event: response.completed" in raw_lines
    assert raw_lines[-1] == "data: [DONE]"


def test_responses_endpoint_preserves_encrypted_reasoning_fields(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that Responses-only reasoning fields are passed through unchanged."""
    mock_response = _FakeResponse(
        payload={
            "id": "resp_456",
            "object": "response",
            "model": "gpt-5-mini",
            "output": [],
            "reasoning": {"encrypted_content": "enc_123"},
        },
        usage=_FakeUsage(input_tokens=8, output_tokens=3),
    )

    with patch(
        "any_llm.gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=mock_response
    ) as mock_aresponses:
        response = client.post(
            "/v1/responses",
            json={
                "model": "openai:gpt-5-mini",
                "input": "Solve this carefully",
                "user": test_user["user_id"],
                "reasoning": {"effort": "high"},
                "include": ["reasoning.encrypted_content"],
            },
            headers=master_key_header,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["reasoning"]["encrypted_content"] == "enc_123"

    mock_aresponses.assert_awaited_once()
    assert mock_aresponses.await_args.kwargs["reasoning"] == {"effort": "high"}
    assert mock_aresponses.await_args.kwargs["include"] == ["reasoning.encrypted_content"]


def test_responses_endpoint_bearer_auth(
    client: TestClient,
    api_key_obj: dict[str, Any],
) -> None:
    """Test authentication via standard Bearer token with a regular API key."""
    mock_response = _FakeResponse(
        payload={"id": "resp_123", "object": "response", "model": "gpt-4.1-mini", "output": []},
        usage=_FakeUsage(input_tokens=10, output_tokens=5),
    )

    with patch("any_llm.gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/responses",
            json={"model": "openai:gpt-4.1-mini", "input": "Hello"},
            headers={API_KEY_HEADER: f"Bearer {api_key_obj['key']}"},
        )

    assert response.status_code == 200
    assert response.json()["id"] == "resp_123"


def test_responses_endpoint_provider_error_non_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a non-streaming provider error returns 500."""
    with patch(
        "any_llm.gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Provider unavailable"),
    ):
        response = client.post(
            "/v1/responses",
            json={"model": "openai:gpt-4.1-mini", "input": "Hello", "user": test_user["user_id"]},
            headers=master_key_header,
        )

    assert response.status_code == 500
    assert "provider" in response.json()["detail"].lower()


def test_responses_endpoint_provider_error_streaming(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a streaming provider creation error returns 500."""
    with patch(
        "any_llm.gateway.api.routes.responses.aresponses",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Provider unavailable"),
    ):
        response = client.post(
            "/v1/responses",
            json={
                "model": "openai:gpt-4.1-mini",
                "input": "Hello",
                "user": test_user["user_id"],
                "stream": True,
            },
            headers=master_key_header,
        )

    assert response.status_code == 500
    assert "provider" in response.json()["detail"].lower()


def test_responses_endpoint_no_usage_data(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test non-streaming response when the provider returns no usage data."""
    mock_response = _FakeResponse(
        payload={"id": "resp_789", "object": "response", "model": "gpt-4.1-mini", "output": []},
        usage=None,
    )

    with patch("any_llm.gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/responses",
            json={"model": "openai:gpt-4.1-mini", "input": "Hello", "user": test_user["user_id"]},
            headers=master_key_header,
        )

    assert response.status_code == 200
    assert response.json()["id"] == "resp_789"


def test_responses_endpoint_streaming_mid_stream_error(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that a mid-stream error emits an SSE error event."""

    async def mock_stream() -> AsyncIterator[_FakeStreamEvent]:
        yield _FakeStreamEvent(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hi"},
        )
        msg = "Stream interrupted"
        raise RuntimeError(msg)

    with patch("any_llm.gateway.api.routes.responses.aresponses", new_callable=AsyncMock, return_value=mock_stream()):
        response = client.post(
            "/v1/responses",
            json={
                "model": "openai:gpt-4.1-mini",
                "input": "Hello",
                "user": test_user["user_id"],
                "stream": True,
            },
            headers=master_key_header,
        )

    assert response.status_code == 200
    raw = response.text
    assert "event: error" in raw
    assert "server_error" in raw
