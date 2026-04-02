"""Tests for environment variable resolution in config."""

import os

import pytest

from any_llm.gateway.core.config import _resolve_env_vars


def test_resolve_existing_env_var() -> None:
    """Test that existing env vars are resolved."""
    os.environ["TEST_RESOLVE_VAR"] = "resolved_value"
    try:
        result = _resolve_env_vars({"key": "${TEST_RESOLVE_VAR}"})
        assert result["key"] == "resolved_value"
    finally:
        del os.environ["TEST_RESOLVE_VAR"]


def test_resolve_missing_env_var_raises() -> None:
    """Test that missing env vars raise ValueError instead of using placeholder."""
    # Ensure the var does not exist
    os.environ.pop("DEFINITELY_MISSING_VAR", None)
    with pytest.raises(ValueError, match="DEFINITELY_MISSING_VAR"):
        _resolve_env_vars({"key": "${DEFINITELY_MISSING_VAR}"})


def test_resolve_nested_dict() -> None:
    """Test that env vars are resolved recursively in nested dicts."""
    os.environ["TEST_NESTED_VAR"] = "nested_value"
    try:
        result = _resolve_env_vars({"outer": {"inner": "${TEST_NESTED_VAR}"}})
        assert result["outer"]["inner"] == "nested_value"
    finally:
        del os.environ["TEST_NESTED_VAR"]


def test_resolve_list_values() -> None:
    """Test that env vars are resolved in list values."""
    os.environ["TEST_LIST_VAR"] = "list_value"
    try:
        result = _resolve_env_vars({"items": ["${TEST_LIST_VAR}", "literal"]})
        assert result["items"] == ["list_value", "literal"]
    finally:
        del os.environ["TEST_LIST_VAR"]


def test_non_env_var_strings_pass_through() -> None:
    """Test that strings not matching ${...} pattern pass through unchanged."""
    result = _resolve_env_vars({"key": "just a string"})
    assert result["key"] == "just a string"


def test_partial_env_var_syntax_passes_through() -> None:
    """Test that partial env var syntax (not matching ${...}) passes through."""
    result = _resolve_env_vars({"key": "${PARTIAL"})
    assert result["key"] == "${PARTIAL"


def test_inline_substitution() -> None:
    """Test that env vars embedded in a larger string are substituted."""
    os.environ["TEST_DB_USER"] = "admin"
    os.environ["TEST_DB_ROLE"] = "readwrite"
    os.environ["TEST_DB_HOST"] = "db.example.com"
    try:
        result = _resolve_env_vars({"url": "postgresql://${TEST_DB_USER}:${TEST_DB_ROLE}@${TEST_DB_HOST}/mydb"})
        assert result["url"] == "postgresql://admin:readwrite@db.example.com/mydb"
    finally:
        del os.environ["TEST_DB_USER"]
        del os.environ["TEST_DB_ROLE"]
        del os.environ["TEST_DB_HOST"]


def test_inline_substitution_missing_var_raises() -> None:
    """Test that a missing var in an inline string raises ValueError."""
    os.environ["TEST_INLINE_OK"] = "present"
    os.environ.pop("TEST_INLINE_MISSING", None)
    try:
        with pytest.raises(ValueError, match="TEST_INLINE_MISSING"):
            _resolve_env_vars({"url": "prefix-${TEST_INLINE_OK}-${TEST_INLINE_MISSING}-suffix"})
    finally:
        del os.environ["TEST_INLINE_OK"]


def test_single_inline_var_with_surrounding_text() -> None:
    """Test a single var reference with surrounding text."""
    os.environ["TEST_PORT"] = "5432"
    try:
        result = _resolve_env_vars({"host": "localhost:${TEST_PORT}"})
        assert result["host"] == "localhost:5432"
    finally:
        del os.environ["TEST_PORT"]
