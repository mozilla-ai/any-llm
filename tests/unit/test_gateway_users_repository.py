from unittest.mock import MagicMock

from any_llm.gateway.repositories.users_repository import get_active_user


def test_get_active_user_skips_for_update_on_sqlite() -> None:
    db = MagicMock()
    db.bind.dialect.name = "sqlite"
    query = db.query.return_value
    query.filter.return_value = query
    query.first.return_value = object()

    get_active_user(db, "user-1", for_update=True)

    query.with_for_update.assert_not_called()


def test_get_active_user_uses_for_update_on_non_sqlite() -> None:
    db = MagicMock()
    db.bind.dialect.name = "postgresql"
    query = db.query.return_value
    query.filter.return_value = query
    query.with_for_update.return_value = query
    query.first.return_value = object()

    get_active_user(db, "user-1", for_update=True)

    query.with_for_update.assert_called_once_with()


def test_get_active_user_uses_for_update_when_bind_is_missing() -> None:
    db = MagicMock()
    db.bind = None
    query = db.query.return_value
    query.filter.return_value = query
    query.with_for_update.return_value = query
    query.first.return_value = object()

    get_active_user(db, "user-1", for_update=True)

    query.with_for_update.assert_called_once_with()
