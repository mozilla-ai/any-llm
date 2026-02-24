"""Tests for pagination parameter bounds on list endpoints."""

from fastapi.testclient import TestClient


def test_users_list_rejects_excessive_limit(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that /v1/users rejects limit > 1000."""
    response = client.get("/v1/users?limit=10000", headers=master_key_header)
    assert response.status_code == 422


def test_users_list_rejects_negative_skip(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that /v1/users rejects negative skip."""
    response = client.get("/v1/users?skip=-1", headers=master_key_header)
    assert response.status_code == 422


def test_users_list_rejects_zero_limit(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that /v1/users rejects limit=0."""
    response = client.get("/v1/users?limit=0", headers=master_key_header)
    assert response.status_code == 422


def test_keys_list_rejects_excessive_limit(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that /v1/keys rejects limit > 1000."""
    response = client.get("/v1/keys?limit=10000", headers=master_key_header)
    assert response.status_code == 422


def test_budgets_list_rejects_excessive_limit(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that /v1/budgets rejects limit > 1000."""
    response = client.get("/v1/budgets?limit=10000", headers=master_key_header)
    assert response.status_code == 422


def test_pricing_list_rejects_excessive_limit(client: TestClient) -> None:
    """Test that /v1/pricing rejects limit > 1000."""
    response = client.get("/v1/pricing?limit=10000")
    assert response.status_code == 422


def test_users_list_accepts_valid_bounds(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that valid pagination parameters are accepted."""
    response = client.get("/v1/users?skip=0&limit=100", headers=master_key_header)
    assert response.status_code == 200

    response = client.get("/v1/users?skip=0&limit=1000", headers=master_key_header)
    assert response.status_code == 200
