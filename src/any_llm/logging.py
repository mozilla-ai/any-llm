import logging
from typing import Any

logger = logging.getLogger("any_llm")


def setup_logger(
    level: int = logging.WARNING,
    rich_tracebacks: bool = True,
    log_format: str | None = None,
    propagate: bool = True,
    reset: bool = False,
    **kwargs: Any,
) -> None:
    """Configure the any_llm logger with the specified settings.

    Args:
        level: The logging level to use (default: logging.WARNING)
        rich_tracebacks: Whether to enable rich tracebacks (default: True)
        log_format: Optional custom log format string
        propagate: Whether to propagate logs to parent loggers (default: True)
        reset: If True, remove existing handlers before attaching a new one (default: False)
        **kwargs: Additional keyword arguments to pass to RichHandler

    """
    logger.setLevel(level)
    logger.propagate = propagate

    if reset:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    try:
        from rich.logging import RichHandler
    except ImportError:
        raise ImportError(
            "rich is required for setup_logger(). "
            "Install it with: pip install 'any-llm-sdk[rich]'"
        ) from None

    handler = RichHandler(rich_tracebacks=rich_tracebacks, markup=True, **kwargs)

    if log_format:
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

    logger.addHandler(handler)


logger.addHandler(logging.NullHandler())
