"""Test scenario definitions and metadata."""

from models import ScenarioID, ScenarioInfo

SCENARIOS: dict[ScenarioID, ScenarioInfo] = {
    ScenarioID.BASIC_COMPLETION: ScenarioInfo(
        id=ScenarioID.BASIC_COMPLETION,
        name="Basic Completion",
        description="Validates a simple chat completion request with required fields only.",
        model_name="test-basic",
        required_fields=["model", "messages"],
        optional_fields=["temperature", "max_tokens", "n"],
    ),
    ScenarioID.TOOL_CALLS: ScenarioInfo(
        id=ScenarioID.TOOL_CALLS,
        name="Tool Calls",
        description="Validates a request that includes tool definitions for function calling.",
        model_name="test-tools",
        required_fields=["model", "messages", "tools"],
        optional_fields=["tool_choice", "parallel_tool_calls"],
    ),
    ScenarioID.TOOL_RESPONSE: ScenarioInfo(
        id=ScenarioID.TOOL_RESPONSE,
        name="Tool Response",
        description="Validates a multi-turn request that includes tool call responses.",
        model_name="test-tool-response",
        required_fields=["model", "messages"],
        optional_fields=["tools"],
    ),
    ScenarioID.STREAMING: ScenarioInfo(
        id=ScenarioID.STREAMING,
        name="Streaming",
        description="Validates a streaming completion request.",
        model_name="test-stream",
        required_fields=["model", "messages", "stream"],
        optional_fields=["stream_options"],
    ),
    ScenarioID.STRUCTURED_OUTPUT: ScenarioInfo(
        id=ScenarioID.STRUCTURED_OUTPUT,
        name="Structured Output",
        description="Validates a request with JSON schema response format.",
        model_name="test-structured",
        required_fields=["model", "messages", "response_format"],
        optional_fields=[],
    ),
    ScenarioID.MULTI_TURN: ScenarioInfo(
        id=ScenarioID.MULTI_TURN,
        name="Multi-turn Conversation",
        description="Validates a multi-turn conversation with proper message ordering.",
        model_name="test-multi-turn",
        required_fields=["model", "messages"],
        optional_fields=[],
    ),
    ScenarioID.SYSTEM_MESSAGE: ScenarioInfo(
        id=ScenarioID.SYSTEM_MESSAGE,
        name="System Message",
        description="Validates proper handling of system messages.",
        model_name="test-system",
        required_fields=["model", "messages"],
        optional_fields=[],
    ),
    ScenarioID.IMAGE_CONTENT: ScenarioInfo(
        id=ScenarioID.IMAGE_CONTENT,
        name="Image Content",
        description="Validates requests with image content in messages.",
        model_name="test-image",
        required_fields=["model", "messages"],
        optional_fields=[],
    ),
    ScenarioID.TEMPERATURE_PARAMS: ScenarioInfo(
        id=ScenarioID.TEMPERATURE_PARAMS,
        name="Temperature Parameters",
        description="Validates generation parameters like temperature, top_p, etc.",
        model_name="test-params",
        required_fields=["model", "messages"],
        optional_fields=["temperature", "top_p", "max_tokens", "max_completion_tokens", "presence_penalty", "frequency_penalty"],
    ),
}


def get_scenario_for_model(model: str) -> ScenarioID | None:
    """Get the scenario ID for a given model name."""
    from models import MODEL_TO_SCENARIO

    return MODEL_TO_SCENARIO.get(model)


def get_all_scenarios() -> list[ScenarioInfo]:
    """Get all available scenarios."""
    return list(SCENARIOS.values())


def get_scenario_info(scenario_id: ScenarioID) -> ScenarioInfo | None:
    """Get info for a specific scenario."""
    return SCENARIOS.get(scenario_id)
