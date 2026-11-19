import asyncio
import time
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
async def test_async_streaming_does_not_block_event_loop() -> None:
    """Reading the SageMaker StreamingBody is blocking I/O, so it must not run on the event loop."""

    class _BlockingStream:
        def __init__(self) -> None:
            self._remaining = 3

        def __iter__(self) -> "_BlockingStream":
            return self

        def __next__(self) -> str:
            if self._remaining == 0:
                raise StopIteration
            self._remaining -= 1
            time.sleep(0.2)
            return "chunk"

    provider = object.__new__(SagemakerProvider)
    ticks = 0
    done = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    params = CompletionParams(model_id="model-id", messages=[{"role": "user", "content": "hi"}], stream=True)
    with patch.object(SagemakerProvider, "_completion", return_value=_BlockingStream()):
        task = asyncio.create_task(ticker())
        stream = await provider._acompletion(params)
        chunks = [chunk async for chunk in stream]  # type: ignore[union-attr]
        done.set()
        await task

    assert chunks == ["chunk"] * 3
    # While the stream blocks for ~0.6s the loop must keep turning (a blocked loop yields ~0 ticks).
    assert ticks >= 10
