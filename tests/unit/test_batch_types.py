from any_llm.exceptions import BatchNotCompleteError
from any_llm.types.batch import BatchResult, BatchResultError, BatchResultItem


def test_batch_result_error_construction() -> None:
    error = BatchResultError(code="invalid_request", message="Bad input")
    assert error.code == "invalid_request"
    assert error.message == "Bad input"


def test_batch_result_item_with_result() -> None:
    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage

    completion = ChatCompletion(
        id="chatcmpl-1",
        model="gpt-4",
        created=1700000000,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello"),
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )
    item = BatchResultItem(custom_id="req-1", result=completion)
    assert item.custom_id == "req-1"
    assert item.result is not None
    assert item.result.id == "chatcmpl-1"
    assert item.error is None


def test_batch_result_item_with_error() -> None:
    error = BatchResultError(code="rate_limit", message="Rate limit exceeded")
    item = BatchResultItem(custom_id="req-2", error=error)
    assert item.custom_id == "req-2"
    assert item.result is None
    assert item.error is not None
    assert item.error.code == "rate_limit"


def test_batch_result_item_defaults() -> None:
    item = BatchResultItem(custom_id="req-3")
    assert item.result is None
    assert item.error is None


def test_batch_result_construction() -> None:
    items = [
        BatchResultItem(custom_id="req-1"),
        BatchResultItem(custom_id="req-2"),
    ]
    result = BatchResult(results=items)
    assert len(result.results) == 2
    assert result.results[0].custom_id == "req-1"


def test_batch_result_empty() -> None:
    result = BatchResult()
    assert result.results == []


def test_batch_not_complete_error_message() -> None:
    error = BatchNotCompleteError(batch_id="batch-123", status="in_progress")
    assert "batch-123" in str(error)
    assert "in_progress" in str(error)
    assert "retrieve_batch()" in str(error)
    assert error.batch_id == "batch-123"
    assert error.batch_status == "in_progress"


def test_batch_not_complete_error_with_provider() -> None:
    error = BatchNotCompleteError(batch_id="batch-456", status="validating", provider_name="openai")
    assert "[openai]" in str(error)
    assert "batch-456" in str(error)
    assert error.provider_name == "openai"
