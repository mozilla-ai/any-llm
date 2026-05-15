import logging

from any_llm.logging import logger


def test_logger_has_null_handler_at_import() -> None:
    """Test that the library logger uses NullHandler by default.

    Libraries should use NullHandler so that importing the library does not
    configure logging for the application. Users can then configure their own
    handlers if they want to see log output.
    See: https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library
    """
    null_handlers = [h for h in logger.handlers if isinstance(h, logging.NullHandler)]
    assert len(null_handlers) >= 1


def test_logger_does_not_install_rich_handler_at_import() -> None:
    """Test that importing the library does not install RichHandler on the logger."""
    from rich.logging import RichHandler

    rich_handlers = [h for h in logger.handlers if isinstance(h, RichHandler)]
    assert len(rich_handlers) == 0


def test_logger_name_is_any_llm() -> None:
    """Test that the logger name matches the library package name."""
    assert logger.name == "any_llm"
