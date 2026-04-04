"""Tests for the bulk GET /v1/usage endpoint."""

import uuid
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from any_llm.gateway.models.entities import UsageLog, User


def _ensure_user(db: Session, user_id: str) -> None:
    if db.query(User).filter(User.user_id == user_id).first() is None:
        db.add(User(user_id=user_id, alias=user_id, spend=0.0, blocked=False))
        db.flush()


def _make_log(
    db: Session,
    *,
    user_id: str,
    timestamp: datetime,
    model: str = "gpt-4",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    cost: float = 0.001,
) -> UsageLog:
    _ensure_user(db, user_id)
    log = UsageLog(
        id=str(uuid.uuid4()),
        user_id=user_id,
        api_key_id=None,
        timestamp=timestamp,
        model=model,
        provider="openai",
        endpoint="/v1/chat/completions",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost=cost,
        status="success",
        error_message=None,
    )
    db.add(log)
    return log


def test_list_usage_requires_master_key(client: TestClient) -> None:
    """Endpoint should reject requests without the master key."""
    response = client.get("/v1/usage")
    assert response.status_code == 401


def test_list_usage_returns_empty_when_no_logs(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Endpoint returns an empty list when no logs exist."""
    response = client.get("/v1/usage", headers=master_key_header)
    assert response.status_code == 200
    assert response.json() == []


def test_list_usage_orders_newest_first(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Logs are returned ordered by timestamp DESC."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    _make_log(db_session, user_id="user-1", timestamp=base)
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=1))
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=2))
    db_session.commit()

    response = client.get("/v1/usage", headers=master_key_header)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Newest first
    timestamps = [entry["timestamp"] for entry in data]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_usage_filter_by_start_date(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """start_date filters out earlier logs."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    _make_log(db_session, user_id="user-1", timestamp=base)
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=2))
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=4))
    db_session.commit()

    response = client.get(
        "/v1/usage",
        params={"start_date": (base + timedelta(hours=1)).isoformat()},
        headers=master_key_header,
    )
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_list_usage_filter_by_end_date(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """end_date is exclusive (half-open window)."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    _make_log(db_session, user_id="user-1", timestamp=base)
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=1))
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=2))
    db_session.commit()

    response = client.get(
        "/v1/usage",
        params={"end_date": (base + timedelta(hours=2)).isoformat()},
        headers=master_key_header,
    )
    assert response.status_code == 200
    # end_date is exclusive, so the 12:00 and 13:00 logs match, 14:00 does not
    assert len(response.json()) == 2


def test_list_usage_filter_by_time_range(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Combined start_date + end_date defines a half-open window [start, end)."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=i))
    db_session.commit()

    response = client.get(
        "/v1/usage",
        params={
            "start_date": (base + timedelta(hours=1)).isoformat(),
            "end_date": (base + timedelta(hours=4)).isoformat(),
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    # [13:00, 14:00, 15:00] — 16:00 excluded by end_date
    assert len(response.json()) == 3


def test_list_usage_filter_by_user_id(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """user_id narrows results to a single user."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    _make_log(db_session, user_id="user-1", timestamp=base)
    _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=1))
    _make_log(db_session, user_id="user-2", timestamp=base + timedelta(hours=2))
    db_session.commit()

    response = client.get(
        "/v1/usage",
        params={"user_id": "user-1"},
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(entry["user_id"] == "user-1" for entry in data)


def test_list_usage_pagination(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """skip and limit paginate consistently."""
    base = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    for i in range(5):
        _make_log(db_session, user_id="user-1", timestamp=base + timedelta(hours=i))
    db_session.commit()

    # First page
    page1 = client.get("/v1/usage", params={"limit": 2}, headers=master_key_header).json()
    assert len(page1) == 2

    # Second page
    page2 = client.get("/v1/usage", params={"skip": 2, "limit": 2}, headers=master_key_header).json()
    assert len(page2) == 2

    # No overlap
    ids1 = {entry["id"] for entry in page1}
    ids2 = {entry["id"] for entry in page2}
    assert ids1.isdisjoint(ids2)


def test_list_usage_response_shape(
    client: TestClient,
    master_key_header: dict[str, str],
    db_session: Session,
) -> None:
    """Response contains all documented fields."""
    _make_log(
        db_session,
        user_id="user-1",
        timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC),
        model="gpt-4",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.0045,
    )
    db_session.commit()

    response = client.get("/v1/usage", headers=master_key_header)
    entry = response.json()[0]
    assert set(entry.keys()) == {
        "id",
        "user_id",
        "api_key_id",
        "timestamp",
        "model",
        "provider",
        "endpoint",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost",
        "status",
        "error_message",
    }
    assert entry["model"] == "gpt-4"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["total_tokens"] == 150
    assert entry["cost"] == 0.0045


def test_list_usage_limit_max_enforced(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """limit > 1000 should return 422."""
    response = client.get("/v1/usage", params={"limit": 2000}, headers=master_key_header)
    assert response.status_code == 422


def test_list_usage_skip_negative_rejected(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Negative skip should return 422."""
    response = client.get("/v1/usage", params={"skip": -1}, headers=master_key_header)
    assert response.status_code == 422
