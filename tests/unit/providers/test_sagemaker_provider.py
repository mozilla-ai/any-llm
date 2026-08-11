from any_llm.providers.sagemaker.sagemaker import SagemakerProvider


def test_per_request_timeout_is_declared_unsupported() -> None:
    """SageMaker serializes completion kwargs into the request body, so the base class rejects a timeout."""
    assert SagemakerProvider.TIMEOUT_SUPPORT == "unsupported"
