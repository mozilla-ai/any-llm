from unittest.mock import patch

from any_llm.providers.sagemaker.sagemaker import SagemakerProvider
from any_llm.types.completion import CompletionParams


def test_timeout_is_dropped_with_a_warning() -> None:
    """SageMaker serializes these kwargs into the request body, so a per-request timeout can't be honored."""
    with patch("any_llm.providers.sagemaker.sagemaker.logger") as mock_logger:
        result = SagemakerProvider._convert_completion_params(
            CompletionParams(model_id="model", messages=[{"role": "user", "content": "Hello"}]),
            timeout=600,
        )

    assert "timeout" not in result
    mock_logger.warning.assert_called_once()
    assert "client_args" in mock_logger.warning.call_args[0][0]
