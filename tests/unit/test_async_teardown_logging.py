import logging

from tests.conftest import _IgnoreEventLoopClosedFilter


def _make_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="asyncio",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_filter_suppresses_event_loop_closed_records() -> None:
    log_filter = _IgnoreEventLoopClosedFilter()
    record = _make_record(
        "Task exception was never retrieved\n"
        "future: <Task finished coro=<AsyncClient.aclose()> exception=RuntimeError('Event loop is closed')>"
    )
    assert log_filter.filter(record) is False


def test_filter_passes_unrelated_asyncio_records() -> None:
    log_filter = _IgnoreEventLoopClosedFilter()
    record = _make_record("Some unrelated asyncio error")
    assert log_filter.filter(record) is True
