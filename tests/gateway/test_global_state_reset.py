"""Tests for global state reset functions."""

import pytest

from any_llm.gateway.auth.dependencies import get_config, reset_config, set_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db.session import reset_db


def test_reset_config_clears_state() -> None:
    """Test that reset_config clears the global config."""
    config = GatewayConfig(
        database_url="postgresql://localhost/test",
        master_key="test",
    )
    set_config(config)
    assert get_config() is config

    reset_config()

    with pytest.raises(RuntimeError, match="Config not initialized"):
        get_config()


def test_reset_db_allows_reinit() -> None:
    """Test that reset_db clears state so init_db can be called again.

    We can't fully test init_db without a database, but we can verify
    reset_db doesn't raise and clears the module state.
    """
    from any_llm.gateway.db import session

    # Verify the function exists and runs without error when nothing is initialized
    reset_db()

    assert session._engine is None
    assert session._SessionLocal is None
