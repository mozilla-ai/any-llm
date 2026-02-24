"""Tests for preserving usage and budget reset logs when deleting users."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db.models import BudgetResetLog, UsageLog
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
    """Deleting a user nullifies user_id and api_key_id in usage logs instead of deleting them."""
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
    assert log_after.user_id is None, "user_id should be NULL after user deletion"
    assert log_after.api_key_id is None, "api_key_id should be NULL (api key cascade-deleted with user)"
    assert log_after.model is not None, "Usage log data should be preserved"


def test_delete_user_preserves_budget_reset_logs(
    client: TestClient,
    master_key_header: dict[str, str],
    api_key_header: dict[str, str],
    test_messages: list[dict[str, str]],
    db_session: Session,
) -> None:
    """Deleting a user nullifies user_id in budget reset logs instead of deleting them."""
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
    assert reset_log_after.user_id is None, "user_id should be NULL after user deletion"
    assert reset_log_after.previous_spend is not None, "Reset log data should be preserved"
