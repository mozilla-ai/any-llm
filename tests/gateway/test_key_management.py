from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi.testclient import TestClient

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from tests.gateway.conftest import MODEL_NAME


def test_create_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating an API key."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "test-key"},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert "id" in data
    assert "key" in data
    assert data["key"].startswith("gw-")
    assert data["key_name"] == "test-key"
    assert data["is_active"] is True
    assert "created_at" in data


def test_create_api_key_with_expiration(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating an API key with expiration."""
    expires_at = (datetime.now(UTC) + timedelta(days=30)).isoformat()

    response = client.post(
        "/v1/keys",
        json={"key_name": "expiring-key", "expires_at": expires_at},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["expires_at"] is not None


def test_create_api_key_with_metadata(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating an API key with metadata."""
    metadata = {"team": "engineering", "environment": "production"}

    response = client.post(
        "/v1/keys",
        json={"key_name": "metadata-key", "metadata": metadata},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["metadata"] == metadata


def test_create_api_key_without_master_key_fails(client: TestClient) -> None:
    """Test that creating key without master key fails."""
    response = client.post(
        "/v1/keys",
        json={"key_name": "test-key"},
    )

    assert response.status_code in [401, 422]


def test_list_api_keys(client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]) -> None:
    """Test listing API keys."""
    response = client.get("/v1/keys", headers=master_key_header)

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
    assert len(data) > 0
    assert any(key["id"] == api_key_obj["id"] for key in data)


def test_list_api_keys_pagination(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test API key listing with pagination."""
    for i in range(5):
        client.post(
            "/v1/keys",
            json={"key_name": f"key-{i}"},
            headers=master_key_header,
        )

    response = client.get("/v1/keys?skip=0&limit=3", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 3


def test_get_api_key(client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]) -> None:
    """Test getting specific API key details."""
    response = client.get(f"/v1/keys/{api_key_obj['id']}", headers=master_key_header)

    assert response.status_code == 200
    data = response.json()

    assert data["id"] == api_key_obj["id"]
    assert data["key_name"] == api_key_obj["key_name"]
    assert "key" not in data


def test_get_nonexistent_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test getting non-existent API key returns 404."""
    response = client.get("/v1/keys/nonexistent-id", headers=master_key_header)

    assert response.status_code == 404


def test_update_api_key(client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]) -> None:
    """Test updating an API key."""
    response = client.patch(
        f"/v1/keys/{api_key_obj['id']}",
        json={"key_name": "updated-key", "is_active": False},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["key_name"] == "updated-key"
    assert data["is_active"] is False


def test_update_api_key_metadata(
    client: TestClient, master_key_header: dict[str, str], api_key_obj: dict[str, Any]
) -> None:
    """Test updating API key metadata."""
    new_metadata = {"updated": True, "version": 2}

    response = client.patch(
        f"/v1/keys/{api_key_obj['id']}",
        json={"metadata": new_metadata},
        headers=master_key_header,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["metadata"] == new_metadata


def test_delete_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test deleting an API key."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "delete-me"},
        headers=master_key_header,
    )
    key_id = create_response.json()["id"]

    delete_response = client.delete(f"/v1/keys/{key_id}", headers=master_key_header)
    assert delete_response.status_code == 204

    get_response = client.get(f"/v1/keys/{key_id}", headers=master_key_header)
    assert get_response.status_code == 404


def test_delete_nonexistent_api_key(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test deleting non-existent API key returns 404."""
    response = client.delete("/v1/keys/nonexistent-id", headers=master_key_header)

    assert response.status_code == 404


def test_api_key_last_used_tracking(
    client: TestClient, master_key_header: dict[str, str], test_config: GatewayConfig
) -> None:
    """Test that last_used_at is updated when key is used."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "usage-tracking"},
        headers=master_key_header,
    )
    api_key = create_response.json()

    get_response = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    initial_last_used = get_response.json()["last_used_at"]

    import time

    time.sleep(0.1)

    client.post(
        "/v1/users",
        json={"user_id": "test-tracking-user"},
        headers=master_key_header,
    )

    _completion_response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "test-tracking-user",
        },
        headers={API_KEY_HEADER: f"Bearer {api_key['key']}"},
    )

    get_response = client.get(f"/v1/keys/{api_key['id']}", headers=master_key_header)
    updated_last_used = get_response.json()["last_used_at"]

    assert updated_last_used is not None
    assert updated_last_used != initial_last_used


def test_inactive_api_key_rejected(
    client: TestClient, master_key_header: dict[str, str], test_config: GatewayConfig
) -> None:
    """Test that inactive API keys are rejected."""
    create_response = client.post(
        "/v1/keys",
        json={"key_name": "inactive-test"},
        headers=master_key_header,
    )
    api_key = create_response.json()

    client.post(
        "/v1/users",
        json={"user_id": "test-inactive-user"},
        headers=master_key_header,
    )

    client.patch(
        f"/v1/keys/{api_key['id']}",
        json={"is_active": False},
        headers=master_key_header,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "test-inactive-user",
        },
        headers={API_KEY_HEADER: f"Bearer {api_key['key']}"},
    )
    assert response.status_code == 401


def test_authorization_header_accepted(client: TestClient, test_config: GatewayConfig) -> None:
    """Test that Authorization header works as fallback for OpenAI client compatibility."""
    # Use Authorization header instead of X-AnyLLM-Key
    auth_header = {"Authorization": f"Bearer {test_config.master_key}"}

    response = client.post(
        "/v1/keys",
        json={"key_name": "auth-header-test"},
        headers=auth_header,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["key_name"] == "auth-header-test"
