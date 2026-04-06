"""Tests for the /v1/messages gateway endpoint."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from any_llm.gateway.core.config import API_KEY_HEADER
from any_llm.types.messages import MessageResponse, MessageUsage, TextBlock, ToolUseBlock


def _make_message_response(**overrides: Any) -> MessageResponse:
    defaults: dict[str, Any] = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [TextBlock(type="text", text="Hello!")],
        "model": "claude-3-5-sonnet",
        "stop_reason": "end_turn",
        "usage": MessageUsage(input_tokens=10, output_tokens=5),
    }
    defaults.update(overrides)
    return MessageResponse(**defaults)


@pytest.fixture
def messages_request_body() -> dict[str, Any]:
    return {
        "model": "anthropic:claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1024,
    }


def test_messages_endpoint_basic_completion(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test basic non-streaming message completion."""
    mock_response = _make_message_response()
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch("any_llm.gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "Hello!"
    assert data["stop_reason"] == "end_turn"


def test_messages_endpoint_requires_auth(
    client: TestClient,
    messages_request_body: dict[str, Any],
) -> None:
    """Test that the endpoint requires authentication."""
    response = client.post(
        "/v1/messages",
        json=messages_request_body,
    )
    assert response.status_code == 401


def test_messages_endpoint_x_api_key_header(
    client: TestClient,
    api_key_obj: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test authentication via x-api-key header (Anthropic client compatibility)."""
    mock_response = _make_message_response()

    with patch("any_llm.gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers={"x-api-key": api_key_obj["key"]},
        )

    assert response.status_code == 200


def test_messages_endpoint_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
    messages_request_body: dict[str, Any],
) -> None:
    """Test that master key auth requires user_id in metadata."""
    mock_response = _make_message_response()

    with patch("any_llm.gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"]["type"] == "invalid_request_error"


def test_messages_endpoint_validation_error(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that validation errors are returned for missing fields."""
    response = client.post(
        "/v1/messages",
        json={"model": "anthropic:claude-3-5-sonnet"},
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_messages_endpoint_with_tools(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test message completion with tools."""
    tool_use_response = _make_message_response(
        content=[
            ToolUseBlock(
                type="tool_use",
                id="toolu_123",
                name="get_weather",
                input={"city": "London"},
            )
        ],
        stop_reason="tool_use",
    )

    request_body = {
        "model": "anthropic:claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "max_tokens": 1024,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
        "metadata": {"user_id": "test-user"},
    }

    with patch("any_llm.gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=tool_use_response):
        response = client.post(
            "/v1/messages",
            json=request_body,
            headers=master_key_header,
        )

    assert response.status_code == 200
    data = response.json()
    assert data["stop_reason"] == "tool_use"
    assert data["content"][0]["type"] == "tool_use"
    assert data["content"][0]["name"] == "get_weather"


def test_messages_endpoint_provider_error_format(
    client: TestClient,
    master_key_header: dict[str, str],
    test_user: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test that provider errors are returned in Anthropic error format."""
    messages_request_body["metadata"] = {"user_id": "test-user"}

    with patch(
        "any_llm.gateway.api.routes.messages.amessages",
        new_callable=AsyncMock,
        side_effect=RuntimeError("Provider unavailable"),
    ):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers=master_key_header,
        )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail["type"] == "error"
    assert detail["error"]["type"] == "api_error"


def test_messages_endpoint_bearer_auth(
    client: TestClient,
    api_key_obj: dict[str, Any],
    messages_request_body: dict[str, Any],
) -> None:
    """Test authentication via standard Bearer token."""
    mock_response = _make_message_response()

    with patch("any_llm.gateway.api.routes.messages.amessages", new_callable=AsyncMock, return_value=mock_response):
        response = client.post(
            "/v1/messages",
            json=messages_request_body,
            headers={API_KEY_HEADER: f"Bearer {api_key_obj['key']}"},
        )

    assert response.status_code == 200
