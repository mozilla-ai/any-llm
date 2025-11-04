from typing import Any

from fastapi.testclient import TestClient

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig


def test_create_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a new user."""
    response = client.post(
        "/v1/users",
        json={"user_id": "test-user-1", "alias": "Test User"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test-user-1"
    assert data["alias"] == "Test User"
    assert data["spend"] == 0.0
    assert data["blocked"] is False


def test_list_users(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test listing users."""
    client.post(
        "/v1/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )
    client.post(
        "/v1/users",
        json={"user_id": "test-user-2"},
        headers=master_key_header,
    )

    response = client.get("/v1/users", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_get_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test getting a specific user."""
    client.post(
        "/v1/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.get("/v1/users/test-user-1", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test-user-1"


def test_update_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test updating a user."""
    client.post(
        "/v1/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.patch(
        "/v1/users/test-user-1",
        json={"blocked": True, "alias": "Updated User"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["blocked"] is True
    assert data["alias"] == "Updated User"


def test_delete_user(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test deleting a user."""
    client.post(
        "/v1/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.delete("/v1/users/test-user-1", headers=master_key_header)
    assert response.status_code == 204

    response = client.get("/v1/users/test-user-1", headers=master_key_header)
    assert response.status_code == 404


def test_create_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a budget."""
    response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["max_budget"] == 100.0


def test_list_budgets(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test listing budgets."""
    client.post(
        "/v1/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    client.post(
        "/v1/budgets",
        json={"max_budget": 200.0},
        headers=master_key_header,
    )

    response = client.get("/v1/budgets", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_set_model_pricing(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test setting model pricing."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "openai:gpt-4o"
    assert data["input_price_per_million"] == 2.5
    assert data["output_price_per_million"] == 10.0


def test_get_pricing(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test getting model pricing."""
    client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o",
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    response = client.get("/v1/pricing/openai:gpt-4o")
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "openai:gpt-4o"


def test_user_with_budget(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a user with a budget."""
    budget_response = client.post(
        "/v1/budgets",
        json={"max_budget": 50.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    response = client.post(
        "/v1/users",
        json={"user_id": "test-user-1", "budget_id": budget_id},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["budget_id"] == budget_id


def test_blocked_user_cannot_make_requests(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that blocked users cannot make completion requests."""
    client.post(
        "/v1/users",
        json={"user_id": "test-user-1", "blocked": True},
        headers=master_key_header,
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "test-user-1",
        },
        headers=api_key_header,
    )
    assert response.status_code == 403
    assert "blocked" in response.json()["detail"].lower()


def test_user_not_found_with_master_key(
    client: TestClient,
    master_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that master key requests fail when user doesn't exist."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "nonexistent-user",
        },
        headers=master_key_header,
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_api_key_requires_existing_user_if_specified(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that API key requests require user to exist if user field is specified."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": test_messages,
            "user": "nonexistent-explicit-user",
        },
        headers=api_key_header,
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_virtual_key_without_user(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_config: GatewayConfig,
) -> None:
    """Test that API keys create a virtual user at key creation time."""
    expected_user_id = f"apikey-{api_key_obj['id']}"

    user_response = client.get(f"/v1/users/{expected_user_id}", headers=master_key_header)
    assert user_response.status_code == 200
    user = user_response.json()
    assert user["user_id"] == expected_user_id
    assert "Virtual user" in user["alias"]

    api_key_header = {API_KEY_HEADER: f"Bearer {api_key_obj['key']}"}
    client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=api_key_header,
    )


def test_master_key_requires_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that master key requests require user field."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        },
        headers=master_key_header,
    )
    assert response.status_code == 400
    assert "user" in response.json()["detail"].lower()
