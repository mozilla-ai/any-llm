import pytest

from tests.integration.test_agent_loop import _call_tool, get_current_date, get_weather


def test_call_tool_ignores_spurious_model_arguments_for_zero_arg_tool() -> None:
    """Ignore model-generated arguments that a zero-argument tool cannot accept."""
    with pytest.warns(UserWarning, match="Ignoring unexpected arguments for get_current_date: result"):
        assert _call_tool(get_current_date, {"result": "unexpected"}) == "2025-12-18 12:30"


def test_call_tool_preserves_declared_tool_arguments() -> None:
    """Preserve arguments declared by a parameterized tool while filtering extras."""
    with pytest.warns(UserWarning, match="Ignoring unexpected arguments for get_weather: result"):
        assert "Paris" in _call_tool(get_weather, {"location": "Paris", "result": "unexpected"})
