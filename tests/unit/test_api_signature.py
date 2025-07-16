import inspect
from any_llm.api import completion, acompletion


def test_completion_and_acompletion_have_same_signature() -> None:
    """Test that completion and acompletion have identical signatures."""
    completion_sig = inspect.signature(completion)
    acompletion_sig = inspect.signature(acompletion)

    # Compare parameters
    assert completion_sig.parameters == acompletion_sig.parameters, (
        "completion and acompletion should have identical parameters"
    )

    # Compare return annotations
    assert completion_sig.return_annotation == acompletion_sig.return_annotation, (
        "completion and acompletion should have identical return annotations"
    )


def test_completion_and_acompletion_have_same_docstring() -> None:
    """Test that completion and acompletion have identical docstrings."""
    completion_doc = completion.__doc__
    acompletion_doc = acompletion.__doc__

    # Both should have docstrings
    assert completion_doc is not None, "completion should have a docstring"
    assert acompletion_doc is not None, "acompletion should have a docstring"

    # Replace "Create a chat completion" with a generic version for comparison
    # and "asynchronously" to account for the async difference
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

        # Check parameter annotations
        assert completion_param.annotation == acompletion_param.annotation, (
            f"Parameter '{param_name}' should have identical annotations"
        )

        # Check parameter defaults
        assert completion_param.default == acompletion_param.default, (
            f"Parameter '{param_name}' should have identical default values"
        )

        # Check parameter kind (positional, keyword, etc.)
        assert completion_param.kind == acompletion_param.kind, (
            f"Parameter '{param_name}' should have identical parameter kinds"
        )
