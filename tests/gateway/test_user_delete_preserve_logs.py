"""Tests for preserving usage and budget reset logs when soft-deleting users."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db.models import APIKey, BudgetResetLog, UsageLog
from tests.gateway.conftest import MODEL_NAME


@pytest.fixture
def db_session(test_config: GatewayConfig) -> Session:
    """Create a standalone DB session for verifying state outside the test client."""
    engine = create_engine(test_config.database_url)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    try:
        yield session  # type: ignore[misc]
    finally:
        session.close()
        engine.dispose()


def test_delete_user_preserves_usage_logs(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Soft-deleting a user preserves FK links in usage logs; API key deletion nullifies api_key_id."""
    client.post("/v1/users", json={"user_id": "del-user"}, headers=master_key_header)
    key_resp = client.post(
        "/v1/keys",
        json={"key_name": "del-key", "user_id": "del-user"},
        headers=master_key_header,
    )
    api_key = key_resp.json()["key"]

    client.post(
        "/v1/chat/completions",
        json={"model": MODEL_NAME, "messages": [{"role": "user", "content": "Hello"}], "user": "del-user"},
        headers={API_KEY_HEADER: f"Bearer {api_key}"},
    )

    logs_before = db_session.query(UsageLog).filter(UsageLog.user_id == "del-user").all()
    assert len(logs_before) > 0
    log_id = logs_before[0].id
    assert logs_before[0].api_key_id is not None

    response = client.delete("/v1/users/del-user", headers=master_key_header)
    assert response.status_code == 204

    db_session.expire_all()
    log_after = db_session.query(UsageLog).filter(UsageLog.id == log_id).first()
    assert log_after is not None, "Usage log should survive user deletion"
    assert log_after.user_id == "del-user", "user_id FK should be preserved (soft-delete keeps user row)"
    assert log_after.api_key_id is None, "api_key_id should be NULL (api keys hard-deleted on soft-delete)"
    assert log_after.model is not None, "Usage log data should be preserved"


def test_delete_user_preserves_budget_reset_logs(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
    db_session: Session,
) -> None:
    """Soft-deleting a user preserves FK links in budget reset logs."""
    budget_resp = client.post(
        "/v1/budgets",
        json={"max_budget": 100.0, "budget_duration_sec": 60},
        headers=master_key_header,
    )
    budget_id = budget_resp.json()["budget_id"]

    client.post(
        "/v1/pricing",
        json={"model_key": MODEL_NAME, "input_price_per_million": 2.5, "output_price_per_million": 10.0},
        headers=master_key_header,
    )

    initial_time = datetime(2025, 10, 1, 12, 0, 0, tzinfo=UTC)
    with patch("any_llm.gateway.routes.users.datetime") as mock_dt:
        mock_dt.now.return_value = initial_time
        client.post(
            "/v1/users",
            json={"user_id": "reset-user", "budget_id": budget_id},
            headers=master_key_header,
        )

    time_after_reset = initial_time + timedelta(seconds=61)
    with (
        patch("any_llm.gateway.budget.datetime") as mock_dt_budget,
        patch("any_llm.gateway.routes.chat.datetime") as mock_dt_chat,
    ):
        mock_dt_budget.now.return_value = time_after_reset
        mock_dt_chat.now.return_value = time_after_reset
        client.post(
            "/v1/chat/completions",
            json={"model": MODEL_NAME, "messages": test_messages, "user": "reset-user"},
            headers=api_key_header,
        )

    reset_logs_before = db_session.query(BudgetResetLog).filter(BudgetResetLog.user_id == "reset-user").all()
    assert len(reset_logs_before) > 0
    reset_log_id = reset_logs_before[0].id

    response = client.delete("/v1/users/reset-user", headers=master_key_header)
    assert response.status_code == 204

    db_session.expire_all()
    reset_log_after = db_session.query(BudgetResetLog).filter(BudgetResetLog.id == reset_log_id).first()
    assert reset_log_after is not None, "Budget reset log should survive user deletion"
    assert reset_log_after.user_id == "reset-user", "user_id FK should be preserved (soft-delete keeps user row)"
    assert reset_log_after.previous_spend is not None, "Reset log data should be preserved"


def test_soft_deleted_user_not_in_list(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Soft-deleted users should not appear in GET /v1/users."""
    client.post("/v1/users", json={"user_id": "list-del-user"}, headers=master_key_header)
    client.delete("/v1/users/list-del-user", headers=master_key_header)

    resp = client.get("/v1/users", headers=master_key_header)
    assert resp.status_code == 200
    user_ids = [u["user_id"] for u in resp.json()]
    assert "list-del-user" not in user_ids


def test_soft_deleted_user_returns_404(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """GET and PATCH on a soft-deleted user should return 404."""
    client.post("/v1/users", json={"user_id": "ghost-user"}, headers=master_key_header)
    client.delete("/v1/users/ghost-user", headers=master_key_header)

    get_resp = client.get("/v1/users/ghost-user", headers=master_key_header)
    assert get_resp.status_code == 404

    patch_resp = client.patch(
        "/v1/users/ghost-user",
        json={"alias": "should-fail"},
        headers=master_key_header,
    )
    assert patch_resp.status_code == 404


def test_recreate_soft_deleted_user(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """POST /v1/users with a previously soft-deleted user_id should restore the user with spend=0."""
    client.post(
        "/v1/users",
        json={"user_id": "revive-user", "alias": "Original"},
        headers=master_key_header,
    )
    client.delete("/v1/users/revive-user", headers=master_key_header)

    resp = client.post(
        "/v1/users",
        json={"user_id": "revive-user", "alias": "Restored"},
        headers=master_key_header,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "revive-user"
    assert data["alias"] == "Restored"
    assert data["spend"] == 0.0

    get_resp = client.get("/v1/users/revive-user", headers=master_key_header)
    assert get_resp.status_code == 200


def test_soft_delete_removes_api_keys(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """API keys should be hard-deleted when a user is soft-deleted."""
    client.post("/v1/users", json={"user_id": "key-del-user"}, headers=master_key_header)
    key_resp = client.post(
        "/v1/keys",
        json={"key_name": "doomed-key", "user_id": "key-del-user"},
        headers=master_key_header,
    )
    key_id = key_resp.json()["id"]

    client.delete("/v1/users/key-del-user", headers=master_key_header)

    db_session.expire_all()
    key = db_session.query(APIKey).filter(APIKey.id == key_id).first()
    assert key is None, "API key should be hard-deleted when user is soft-deleted"


def test_cascade_delete_api_keys_on_hard_delete(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Raw SQL DELETE on users should cascade to api_keys via ondelete='CASCADE'."""
    client.post("/v1/users", json={"user_id": "cascade-user"}, headers=master_key_header)
    key_resp = client.post(
        "/v1/keys",
        json={"key_name": "cascade-key", "user_id": "cascade-user"},
        headers=master_key_header,
    )
    key_id = key_resp.json()["id"]

    db_session.expire_all()
    db_session.execute(text("DELETE FROM users WHERE user_id = :uid"), {"uid": "cascade-user"})
    db_session.commit()

    key = db_session.query(APIKey).filter(APIKey.id == key_id).first()
    assert key is None, "API key should be cascade-deleted when user row is hard-deleted"
