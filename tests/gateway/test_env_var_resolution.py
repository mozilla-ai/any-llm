"""Tests for environment variable resolution in config."""

import os

import pytest

from any_llm.gateway.config import _resolve_env_vars


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
