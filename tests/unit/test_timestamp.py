"""Unit tests for any_llm.utils.timestamp."""

from datetime import UTC, datetime

from any_llm.utils.timestamp import to_datetime


def test_to_datetime_returns_none_for_none() -> None:
    assert to_datetime(None) is None


def test_to_datetime_passes_through_datetime() -> None:
    dt = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    assert to_datetime(dt) is dt


def test_to_datetime_converts_epoch_seconds() -> None:
    expected = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    result = to_datetime(int(expected.timestamp()))
    assert result == expected
