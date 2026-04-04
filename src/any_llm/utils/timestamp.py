"""Timestamp coercion helpers."""

from datetime import UTC, datetime


def to_datetime(value: int | datetime | None) -> datetime | None:
    """Coerce an int (Unix epoch seconds) or datetime value to a timezone-aware datetime.

    Returns None if the value is None.
    """
    if value is None:
        return None
    if isinstance(value, int):
        return datetime.fromtimestamp(value, tz=UTC)
    return value
