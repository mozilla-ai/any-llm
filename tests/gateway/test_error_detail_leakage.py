"""Tests for error message sanitization."""

from typing import Any

from fastapi.testclient import TestClient


def test_provider_error_does_not_leak_details(
    client: TestClient,
    api_key_header: dict[str, str],
    test_user: dict[str, Any],
) -> None:
    """Test that provider errors return a generic message without internal details."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hello"}],
            "user": test_user["user_id"],
        },
        headers=api_key_header,
    )
    assert response.status_code == 500
    detail = response.json()["detail"]
    # Should not contain provider-specific error details
    assert "api" not in detail.lower() or "provider" in detail.lower()
    assert "could not be completed" in detail.lower()


def test_health_readiness_does_not_leak_db_details(client: TestClient) -> None:
    """Test that readiness endpoint doesn't leak database details on success."""
    response = client.get("/health/readiness")
    assert response.status_code == 200
    data = response.json()
    # Should not contain "error" key
    assert "error" not in data
