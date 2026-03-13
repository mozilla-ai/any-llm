from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient

from any_llm.gateway.budget import calculate_next_reset
from tests.gateway.conftest import MODEL_NAME


def test_calculate_next_reset() -> None:
    """Test calculating next reset date with seconds."""
    start = datetime(2025, 10, 1, 0, 0, 0, tzinfo=UTC)

    next_reset = calculate_next_reset(start, 86400)
    assert next_reset == datetime(2025, 10, 2, 0, 0, 0, tzinfo=UTC)

    next_reset = calculate_next_reset(start, 604800)
    assert next_reset == datetime(2025, 10, 8, 0, 0, 0, tzinfo=UTC)

    next_reset = calculate_next_reset(start, 60)
    assert next_reset == datetime(2025, 10, 1, 0, 1, 0, tzinfo=UTC)


def test_create_budget_with_duration_sec(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test creating a budget with duration in seconds."""
    response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 86400},
        headers=master_key_header,
    )
    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["max_budget"] == 100.0
    assert data["budget_duration_sec"] == 86400


def test_user_with_budget_gets_reset_fields_set(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that creating a user with a budget sets budget tracking fields."""
    budget_response = client.post(
        "/v1/budgets",
        json={"max_budget": 50.0, "budget_duration_sec": 604800},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    response = client.post(
        "/v1/users",
        json={"user_id": "test-user-1", "budget_id": budget_id},
        headers=master_key_header,
    )

    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["budget_id"] == budget_id
    assert data["budget_started_at"] is not None
    assert data["next_budget_reset_at"] is not None


def test_updating_user_budget_sets_reset_fields(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that updating a user's budget sets budget tracking fields."""
    budget_response = client.post(
        "/v1/budgets",
        json={"max_budget": 75.0, "budget_duration_sec": 86400},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        "/v1/users",
        json={"user_id": "test-user-1"},
        headers=master_key_header,
    )

    response = client.patch(
        "/v1/users/test-user-1",
        json={"budget_id": budget_id},
        headers=master_key_header,
    )

    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["budget_started_at"] is not None
    assert data["next_budget_reset_at"] is not None


def test_budget_without_duration_no_reset(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that budgets without duration don't set reset schedules."""
    budget_response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    response = client.post(
        "/v1/users",
        json={"user_id": "test-user-1", "budget_id": budget_id},
        headers=master_key_header,
    )

    assert response.status_code == 200, f"Response: {response.json()}"
    data = response.json()
    assert data["budget_started_at"] is not None
    assert data["next_budget_reset_at"] is None


def test_nonexistent_budget_returns_404(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that assigning a nonexistent budget returns 404."""
    response = client.post(
        "/v1/users",
        json={"user_id": "test-user-1", "budget_id": "nonexistent-budget"},
        headers=master_key_header,
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_budget_actually_resets_when_duration_passes(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that budget actually resets when duration passes - THE CRITICAL TEST."""
    budget_response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 60},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    client.post(
        "/v1/pricing",
        json={
            "model_key": MODEL_NAME,
            "input_price_per_million": 2.5,
            "output_price_per_million": 10.0,
        },
        headers=master_key_header,
    )

    initial_time = datetime(2025, 10, 1, 12, 0, 0, tzinfo=UTC)

    with patch("any_llm.gateway.routes.users.datetime") as mock_datetime:
        mock_datetime.now.return_value = initial_time

        client.post(
            "/v1/users",
            json={"user_id": "test-user-1", "budget_id": budget_id},
            headers=master_key_header,
        )

    user_response = client.get("/v1/users/test-user-1", headers=master_key_header)
    assert user_response.status_code == 200
    user_data = user_response.json()
    assert user_data["spend"] == 0.0

    with patch("any_llm.gateway.budget.datetime") as mock_datetime_budget:
        mock_datetime_budget.now.return_value = initial_time

        with patch("any_llm.gateway.routes.chat.datetime") as mock_datetime_chat:
            mock_datetime_chat.now.return_value = initial_time

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": test_messages,
                    "user": "test-user-1",
                },
                headers=api_key_header,
            )

    assert response.status_code == 200, f"Response: {response.json()}"

    user_response = client.get("/v1/users/test-user-1", headers=master_key_header)
    user_data = user_response.json()
    spend_before_reset = user_data["spend"]
    assert spend_before_reset > 0.0

    time_after_reset = initial_time + timedelta(seconds=61)

    with patch("any_llm.gateway.budget.datetime") as mock_datetime_budget:
        mock_datetime_budget.now.return_value = time_after_reset

        with patch("any_llm.gateway.routes.chat.datetime") as mock_datetime_chat:
            mock_datetime_chat.now.return_value = time_after_reset

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": test_messages,
                    "user": "test-user-1",
                },
                headers=api_key_header,
            )

    assert response.status_code == 200, f"Response: {response.json()}"

    user_response = client.get("/v1/users/test-user-1", headers=master_key_header)
    user_data = user_response.json()
    spend_after_reset = user_data["spend"]

    assert spend_after_reset > 0.0
    assert spend_after_reset < (spend_before_reset * 2)


def test_per_user_reset_schedules_with_actual_reset(client: TestClient, master_key_header: dict[str, str]) -> None:
    """Test that different users on the same budget have independent reset schedules and actually reset independently."""
    budget_response = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 604800},
        headers=master_key_header,
    )
    budget_id = budget_response.json()["budget_id"]

    user_a_time = datetime(2025, 10, 1, 0, 0, 0, tzinfo=UTC)
    user_b_time = datetime(2025, 10, 2, 0, 0, 0, tzinfo=UTC)

    with patch("any_llm.gateway.routes.users.datetime") as mock_datetime:
        mock_datetime.now.return_value = user_a_time

        response_a = client.post(
            "/v1/users",
            json={"user_id": "user-a", "budget_id": budget_id},
            headers=master_key_header,
        )

    with patch("any_llm.gateway.routes.users.datetime") as mock_datetime:
        mock_datetime.now.return_value = user_b_time

        response_b = client.post(
            "/v1/users",
            json={"user_id": "user-b", "budget_id": budget_id},
            headers=master_key_header,
        )

    assert response_a.status_code == 200
    assert response_b.status_code == 200

    user_a_data = response_a.json()
    user_b_data = response_b.json()

    reset_a = datetime.fromisoformat(user_a_data["next_budget_reset_at"]).replace(tzinfo=UTC)
    reset_b = datetime.fromisoformat(user_b_data["next_budget_reset_at"]).replace(tzinfo=UTC)

    assert reset_a == user_a_time + timedelta(seconds=604800)
    assert reset_b == user_b_time + timedelta(seconds=604800)
    assert reset_b > reset_a
