"""Tests for preserving usage logs when deleting users."""

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db.models import UsageLog
from tests.gateway.conftest import MODEL_NAME


def test_delete_user_preserves_usage_logs(
    client: TestClient,
    master_key_header: dict[str, str],
    test_config: GatewayConfig,
) -> None:
    """Test that deleting a user nullifies user_id in usage logs instead of deleting them."""
    # Create user and API key
    client.post(
        "/v1/users",
        json={"user_id": "deletable-user"},
        headers=master_key_header,
    )
    key_response = client.post(
        "/v1/keys",
        json={"key_name": "deletable-key", "user_id": "deletable-user"},
        headers=master_key_header,
    )
    api_key = key_response.json()["key"]

    # Make a completion request to generate a usage log
    client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello"}],
            "user": "deletable-user",
        },
        headers={API_KEY_HEADER: f"Bearer {api_key}"},
    )

    # Verify usage log exists
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_local()
    try:
        logs_before = db.query(UsageLog).filter(UsageLog.user_id == "deletable-user").all()
        assert len(logs_before) > 0, "Should have at least one usage log before deletion"
        log_id = logs_before[0].id
    finally:
        db.close()

    # Delete the user
    response = client.delete("/v1/users/deletable-user", headers=master_key_header)
    assert response.status_code == 204

    # Verify usage log still exists but with null user_id
    db = session_local()
    try:
        log_after = db.query(UsageLog).filter(UsageLog.id == log_id).first()
        assert log_after is not None, "Usage log should still exist after user deletion"
        assert log_after.user_id is None, "Usage log user_id should be NULL after user deletion"
        assert log_after.model is not None, "Usage log data should be preserved"
    finally:
        db.close()
