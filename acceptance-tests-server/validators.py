"""Request validators for each test scenario."""

from models import (
    ChatCompletionRequest,
    ImageContentPart,
    ScenarioID,
    TextContentPart,
    ValidationError,
    ValidationResult,
)


def validate_basic_completion(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate a basic completion request."""
    errors: list[ValidationError] = []

    if not request.model:
        errors.append(ValidationError(field="model", message="Model is required"))

    if not request.messages:
        errors.append(ValidationError(field="messages", message="Messages array is required"))
    elif len(request.messages) == 0:
        errors.append(ValidationError(field="messages", message="Messages array cannot be empty"))
    else:
        last_message = request.messages[-1]
        if last_message.role not in ("user", "system"):
            errors.append(
                ValidationError(
                    field="messages[-1].role",
                    message="Last message should typically be from user",
                    expected="user",
                    actual=last_message.role,
                )
            )

    return ValidationResult(
        scenario=ScenarioID.BASIC_COMPLETION,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_tool_calls(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate a tool calls request."""
    errors: list[ValidationError] = []

    if not request.tools:
        errors.append(ValidationError(field="tools", message="Tools array is required for this scenario"))
    else:
        for i, tool in enumerate(request.tools):
            if tool.type != "function":
                errors.append(
                    ValidationError(
                        field=f"tools[{i}].type",
                        message="Tool type must be 'function'",
                        expected="function",
                        actual=tool.type,
                    )
                )
            if not tool.function.name:
                errors.append(
                    ValidationError(field=f"tools[{i}].function.name", message="Function name is required")
                )

    if request.tool_choice is not None:
        if isinstance(request.tool_choice, str):
            if request.tool_choice not in ("none", "auto", "required"):
                errors.append(
                    ValidationError(
                        field="tool_choice",
                        message="Invalid tool_choice value",
                        expected="none | auto | required | object",
                        actual=request.tool_choice,
                    )
                )

    return ValidationResult(
        scenario=ScenarioID.TOOL_CALLS,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_tool_response(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate a tool response request (multi-turn with tool results)."""
    errors: list[ValidationError] = []

    has_assistant_with_tool_calls = False
    has_tool_response = False
    tool_call_ids: set[str] = set()
    tool_response_ids: set[str] = set()

    for i, msg in enumerate(request.messages):
        if msg.role == "assistant" and msg.tool_calls:
            has_assistant_with_tool_calls = True
            for tc in msg.tool_calls:
                tool_call_ids.add(tc.id)

        if msg.role == "tool":
            has_tool_response = True
            if not msg.tool_call_id:
                errors.append(
                    ValidationError(
                        field=f"messages[{i}].tool_call_id",
                        message="Tool message must have tool_call_id",
                    )
                )
            else:
                tool_response_ids.add(msg.tool_call_id)

            if msg.content is None:
                errors.append(
                    ValidationError(
                        field=f"messages[{i}].content",
                        message="Tool message must have content",
                    )
                )

    if not has_assistant_with_tool_calls:
        errors.append(
            ValidationError(
                field="messages",
                message="Expected an assistant message with tool_calls before tool responses",
            )
        )

    if not has_tool_response:
        errors.append(
            ValidationError(
                field="messages",
                message="Expected at least one tool response message",
            )
        )

    unmatched_responses = tool_response_ids - tool_call_ids
    if unmatched_responses:
        errors.append(
            ValidationError(
                field="messages",
                message="Tool response IDs don't match any tool call IDs",
                expected=list(tool_call_ids),
                actual=list(unmatched_responses),
            )
        )

    return ValidationResult(
        scenario=ScenarioID.TOOL_RESPONSE,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_streaming(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate a streaming request."""
    errors: list[ValidationError] = []

    if request.stream is not True:
        errors.append(
            ValidationError(
                field="stream",
                message="Stream must be true for streaming scenario",
                expected=True,
                actual=request.stream,
            )
        )

    if request.stream_options is not None:
        if not isinstance(request.stream_options.include_usage, bool | type(None)):
            errors.append(
                ValidationError(
                    field="stream_options.include_usage",
                    message="include_usage must be a boolean or null",
                )
            )

    return ValidationResult(
        scenario=ScenarioID.STREAMING,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_structured_output(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate a structured output request."""
    errors: list[ValidationError] = []

    if not request.response_format:
        errors.append(
            ValidationError(
                field="response_format",
                message="response_format is required for structured output scenario",
            )
        )
    else:
        if request.response_format.type not in ("json_object", "json_schema"):
            errors.append(
                ValidationError(
                    field="response_format.type",
                    message="response_format.type must be 'json_object' or 'json_schema'",
                    expected="json_object | json_schema",
                    actual=request.response_format.type,
                )
            )

        if request.response_format.type == "json_schema":
            if not request.response_format.json_schema:
                errors.append(
                    ValidationError(
                        field="response_format.json_schema",
                        message="json_schema is required when type is 'json_schema'",
                    )
                )
            elif not request.response_format.json_schema.name:
                errors.append(
                    ValidationError(
                        field="response_format.json_schema.name",
                        message="json_schema.name is required",
                    )
                )

    return ValidationResult(
        scenario=ScenarioID.STRUCTURED_OUTPUT,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_multi_turn(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate a multi-turn conversation request."""
    errors: list[ValidationError] = []

    if len(request.messages) < 3:
        errors.append(
            ValidationError(
                field="messages",
                message="Multi-turn conversation should have at least 3 messages",
                expected=">=3",
                actual=len(request.messages),
            )
        )

    has_user = False
    has_assistant = False
    prev_role = None

    for i, msg in enumerate(request.messages):
        if msg.role == "user":
            has_user = True
        if msg.role == "assistant":
            has_assistant = True

        if msg.role == prev_role and msg.role in ("user", "assistant"):
            errors.append(
                ValidationError(
                    field=f"messages[{i}].role",
                    message="Consecutive messages should alternate between user and assistant",
                    expected="alternating roles",
                    actual=f"consecutive {msg.role}",
                )
            )

        if msg.role not in ("system", "tool"):
            prev_role = msg.role

    if not has_user:
        errors.append(ValidationError(field="messages", message="Must have at least one user message"))

    if not has_assistant:
        errors.append(ValidationError(field="messages", message="Must have at least one assistant message"))

    return ValidationResult(
        scenario=ScenarioID.MULTI_TURN,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_system_message(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate system message handling."""
    errors: list[ValidationError] = []

    has_system = False
    system_index = -1

    for i, msg in enumerate(request.messages):
        if msg.role == "system":
            has_system = True
            system_index = i
            break

    if not has_system:
        errors.append(
            ValidationError(
                field="messages",
                message="Expected a system message for this scenario",
            )
        )
    elif system_index != 0:
        errors.append(
            ValidationError(
                field=f"messages[{system_index}]",
                message="System message should be the first message",
                expected=0,
                actual=system_index,
            )
        )

    return ValidationResult(
        scenario=ScenarioID.SYSTEM_MESSAGE,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_image_content(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate image content in messages."""
    errors: list[ValidationError] = []

    has_image = False

    for i, msg in enumerate(request.messages):
        if isinstance(msg.content, list):
            for j, part in enumerate(msg.content):
                if isinstance(part, ImageContentPart):
                    has_image = True
                    if not part.image_url.url:
                        errors.append(
                            ValidationError(
                                field=f"messages[{i}].content[{j}].image_url.url",
                                message="Image URL is required",
                            )
                        )
                    elif not (
                        part.image_url.url.startswith("http://")
                        or part.image_url.url.startswith("https://")
                        or part.image_url.url.startswith("data:")
                    ):
                        errors.append(
                            ValidationError(
                                field=f"messages[{i}].content[{j}].image_url.url",
                                message="Image URL must be http://, https://, or data: URI",
                                actual=part.image_url.url[:50],
                            )
                        )
                elif isinstance(part, TextContentPart):
                    if not part.text:
                        errors.append(
                            ValidationError(
                                field=f"messages[{i}].content[{j}].text",
                                message="Text content cannot be empty",
                            )
                        )

    if not has_image:
        errors.append(
            ValidationError(
                field="messages",
                message="Expected at least one message with image content",
            )
        )

    return ValidationResult(
        scenario=ScenarioID.IMAGE_CONTENT,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


def validate_temperature_params(request: ChatCompletionRequest, request_id: str) -> ValidationResult:
    """Validate temperature and other generation parameters."""
    errors: list[ValidationError] = []

    if request.temperature is not None:
        if not 0.0 <= request.temperature <= 2.0:
            errors.append(
                ValidationError(
                    field="temperature",
                    message="Temperature must be between 0.0 and 2.0",
                    expected="0.0 <= x <= 2.0",
                    actual=request.temperature,
                )
            )

    if request.top_p is not None:
        if not 0.0 <= request.top_p <= 1.0:
            errors.append(
                ValidationError(
                    field="top_p",
                    message="top_p must be between 0.0 and 1.0",
                    expected="0.0 <= x <= 1.0",
                    actual=request.top_p,
                )
            )

    if request.max_tokens is not None:
        if request.max_tokens <= 0:
            errors.append(
                ValidationError(
                    field="max_tokens",
                    message="max_tokens must be positive",
                    expected=">0",
                    actual=request.max_tokens,
                )
            )

    if request.max_completion_tokens is not None:
        if request.max_completion_tokens <= 0:
            errors.append(
                ValidationError(
                    field="max_completion_tokens",
                    message="max_completion_tokens must be positive",
                    expected=">0",
                    actual=request.max_completion_tokens,
                )
            )

    if request.presence_penalty is not None:
        if not -2.0 <= request.presence_penalty <= 2.0:
            errors.append(
                ValidationError(
                    field="presence_penalty",
                    message="presence_penalty must be between -2.0 and 2.0",
                    expected="-2.0 <= x <= 2.0",
                    actual=request.presence_penalty,
                )
            )

    if request.frequency_penalty is not None:
        if not -2.0 <= request.frequency_penalty <= 2.0:
            errors.append(
                ValidationError(
                    field="frequency_penalty",
                    message="frequency_penalty must be between -2.0 and 2.0",
                    expected="-2.0 <= x <= 2.0",
                    actual=request.frequency_penalty,
                )
            )

    return ValidationResult(
        scenario=ScenarioID.TEMPERATURE_PARAMS,
        passed=len(errors) == 0,
        errors=errors,
        request_id=request_id,
    )


VALIDATORS = {
    ScenarioID.BASIC_COMPLETION: validate_basic_completion,
    ScenarioID.TOOL_CALLS: validate_tool_calls,
    ScenarioID.TOOL_RESPONSE: validate_tool_response,
    ScenarioID.STREAMING: validate_streaming,
    ScenarioID.STRUCTURED_OUTPUT: validate_structured_output,
    ScenarioID.MULTI_TURN: validate_multi_turn,
    ScenarioID.SYSTEM_MESSAGE: validate_system_message,
    ScenarioID.IMAGE_CONTENT: validate_image_content,
    ScenarioID.TEMPERATURE_PARAMS: validate_temperature_params,
}


def validate_request(
    scenario: ScenarioID, request: ChatCompletionRequest, request_id: str
) -> ValidationResult:
    """Validate a request against a specific scenario."""
    validator = VALIDATORS.get(scenario)
    if not validator:
        return ValidationResult(
            scenario=scenario,
            passed=False,
            errors=[
                ValidationError(
                    field="scenario", message=f"Unknown scenario: {scenario}"
                )
            ],
            request_id=request_id,
        )
    return validator(request, request_id)
