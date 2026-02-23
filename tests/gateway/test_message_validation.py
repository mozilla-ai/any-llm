"""Tests for chat completion message validation."""

from fastapi.testclient import TestClient


def test_message_without_role_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that messages without a role field are rejected."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"content": "Hello"}],
            "user": "test-user",
        },
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_empty_messages_rejected(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that empty messages list is rejected."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [],
            "user": "test-user",
        },
        headers=master_key_header,
    )
    assert response.status_code == 422


def test_valid_message_with_extra_fields_accepted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that messages with extra provider-specific fields are accepted."""
    # This should pass validation (though may fail at the provider level)
    # The point is the request model accepts it
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "Hello", "name": "test-user-name"}],
            "user": "test-user",
        },
        headers=master_key_header,
    )
    # Will be 404 (user not found) or 500 (provider error), not 422
    assert response.status_code != 422


def test_message_with_null_content_accepted(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that messages with null content (e.g., tool call responses) are accepted."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "assistant", "content": None}],
            "user": "test-user",
        },
        headers=master_key_header,
    )
    # Should not be a validation error
    assert response.status_code != 422
