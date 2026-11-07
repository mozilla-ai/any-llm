import threading
from collections.abc import Iterator
from unittest.mock import patch

import pytest

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


@pytest.mark.asyncio
async def test_async_stream_reads_events_off_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    read_threads: list[int] = []

    def blocking_chunks() -> Iterator[str]:
        for item in ("a", "b"):
            read_threads.append(threading.get_ident())
            yield item

    provider = SagemakerProvider.__new__(SagemakerProvider)
    params = CompletionParams(model_id="endpoint", messages=[{"role": "user", "content": "hi"}], stream=True)

    with patch.object(SagemakerProvider, "_completion", return_value=blocking_chunks()):
        stream = await provider._acompletion(params)
        received = [chunk async for chunk in stream]  # type: ignore[union-attr]

    assert received == ["a", "b"]
    assert loop_thread not in read_threads
