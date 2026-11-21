from any_llm.providers.sagemaker.sagemaker import SagemakerProvider
from any_llm.providers.sagemaker.utils import _convert_params
from any_llm.types.completion import CompletionParams


def test_per_request_timeout_is_declared_unsupported() -> None:
    """SageMaker serializes completion kwargs into the request body, so the base class rejects a timeout."""
    assert SagemakerProvider.TIMEOUT_SUPPORT == "unsupported"


def test_convert_params_forwards_zero_temperature() -> None:
    """temperature=0.0 asks for greedy decoding and must not be dropped as a falsy value."""
    result = _convert_params(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}], temperature=0.0),
        {},
    )

    assert result["temperature"] == 0.0


def test_convert_params_forwards_zero_top_p() -> None:
    """top_p=0.0 is inside the documented range, so it must reach the request body."""
    result = _convert_params(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}], top_p=0.0),
        {},
    )

    assert result["top_p"] == 0.0


def test_convert_params_omits_unset_sampling_params() -> None:
    """Params the caller never set stay absent so the model applies its own defaults."""
    result = _convert_params(
        CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}]),
        {},
    )

    assert "temperature" not in result
    assert "top_p" not in result


def test_convert_params_keeps_system_messages_in_messages() -> None:
    """The OpenAI chat schema served by SageMaker containers reads system prompts from `messages` only."""
    messages = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "Be brief."},
    ]
    result = _convert_params(CompletionParams(model_id="model-id", messages=messages), {})

    assert result["messages"] == messages
    assert "system" not in result
