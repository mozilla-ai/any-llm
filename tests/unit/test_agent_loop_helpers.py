from tests.integration.test_agent_loop import _call_tool, get_current_date, get_weather


def test_call_tool_ignores_spurious_model_arguments_for_zero_arg_tool() -> None:
    assert _call_tool(get_current_date, {"result": "unexpected"}) == "2025-12-18 12:30"


def test_call_tool_preserves_declared_tool_arguments() -> None:
    assert "Paris" in _call_tool(get_weather, {"location": "Paris", "result": "unexpected"})
