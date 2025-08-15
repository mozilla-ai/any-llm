import inspect

from any_llm.api import acompletion, aresponses, completion, responses


def test_completion_and_acompletion_have_same_signature() -> None:
    """Test that completion and acompletion have identical signatures."""
    completion_sig = inspect.signature(completion)
    acompletion_sig = inspect.signature(acompletion)

    assert completion_sig.parameters == acompletion_sig.parameters, (
        "completion and acompletion should have identical parameters"
    )


def test_completion_and_acompletion_have_same_docstring() -> None:
    """Test that completion and acompletion have identical docstrings."""
    completion_doc = completion.__doc__
    acompletion_doc = acompletion.__doc__

    assert completion_doc is not None, "completion should have a docstring"
    assert acompletion_doc is not None, "acompletion should have a docstring"

    normalized_completion_doc = completion_doc.replace("Create a chat completion.", "Create a chat completion")
    normalized_acompletion_doc = acompletion_doc.replace(
        "Create a chat completion asynchronously.", "Create a chat completion"
    )

    assert normalized_completion_doc == normalized_acompletion_doc, (
        "completion and acompletion should have identical docstrings (except for the async-specific description)"
    )


def test_completion_and_acompletion_parameter_details() -> None:
    """Test that completion and acompletion parameters have identical details."""
    completion_sig = inspect.signature(completion)
    acompletion_sig = inspect.signature(acompletion)

    for param_name in completion_sig.parameters:
        completion_param = completion_sig.parameters[param_name]
        acompletion_param = acompletion_sig.parameters[param_name]

        assert completion_param.annotation == acompletion_param.annotation, (
            f"Parameter '{param_name}' should have identical annotations"
        )

        assert completion_param.default == acompletion_param.default, (
            f"Parameter '{param_name}' should have identical default values"
        )

        assert completion_param.kind == acompletion_param.kind, (
            f"Parameter '{param_name}' should have identical parameter kinds"
        )


def test_responses_and_aresponses_have_same_signature() -> None:
    """Test that responses and aresponses have identical signatures."""
    responses_sig = inspect.signature(responses)
    aresponses_sig = inspect.signature(aresponses)

    assert responses_sig.parameters == aresponses_sig.parameters, (
        "responses and aresponses should have identical parameters"
    )


def test_responses_and_aresponses_have_same_docstring() -> None:
    """Test that responses and aresponses have identical docstrings."""
    responses_doc = responses.__doc__
    aresponses_doc = aresponses.__doc__

    assert responses_doc is not None, "responses should have a docstring"
    assert aresponses_doc is not None, "aresponses should have a docstring"

    assert responses_doc == aresponses_doc, "responses and aresponses should have identical docstrings"


def test_responses_and_aresponses_parameter_details() -> None:
    """Test that responses and aresponses parameters have identical details."""
    responses_sig = inspect.signature(responses)
    aresponses_sig = inspect.signature(aresponses)

    for param_name in responses_sig.parameters:
        responses_param = responses_sig.parameters[param_name]
        aresponses_param = aresponses_sig.parameters[param_name]

        assert responses_param.annotation == aresponses_param.annotation, (
            f"Parameter '{param_name}' should have identical annotations"
        )

        assert responses_param.default == aresponses_param.default, (
            f"Parameter '{param_name}' should have identical default values"
        )

        assert responses_param.kind == aresponses_param.kind, (
            f"Parameter '{param_name}' should have identical parameter kinds"
        )
