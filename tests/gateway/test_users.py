from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

def test_get_user_with_token_usage(client: TestClient, db: Session, user_factory, usage_log_factory):
    """Test retrieving a user includes token usage."""
    user = user_factory(user_id="test_user_with_usage")
    usage_log_factory(
        user_id=user.user_id,
        prompt_tokens=100,
        completion_tokens=200,
        total_tokens=300,
    )
    usage_log_factory(
        user_id=user.user_id,
        prompt_tokens=50,
        completion_tokens=150,
        total_tokens=200,
    )

    response = client.get(f"/v1/users/{user.user_id}")

    assert response.status_code == 200

    user_data = response.json()
    assert user_data["total_input_tokens"] == 150
    assert user_data["total_output_tokens"] == 350
    assert user_data["total_tokens"] == 500

def test_get_user_without_token_usage(client: TestClient, db: Session, user_factory):
    """Test retrieving a user with no token usage returns zero values."""
    user = user_factory(user_id="test_user_no_usage")

    response = client.get(f"/v1/users/{user.user_id}")

    assert response.status_code == 200

    user_data = response.json()
    assert user_data["total_input_tokens"] == 0
    assert user_data["total_output_tokens"] == 0
    assert user_data["total_tokens"] == 0
