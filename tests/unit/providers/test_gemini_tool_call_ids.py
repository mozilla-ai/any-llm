import json
from typing import Any

import httpx
import pytest
from google.genai import types

from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.utils import (
    _convert_messages,
    _convert_response_to_response_dict,
    _create_openai_chunk_from_google_chunk,
)
from any_llm.types.completion import ChatCompletion


def _response(call_id: str | None, city: str = "Lisbon") -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(id=call_id, name="get_weather", args={"city": city}),
                            thought_signature=b"signature",
                        )
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )


@pytest.mark.parametrize("call_id", ["call_120847", None, ""])
def test_completion_tool_call_id(call_id: str | None) -> None:
    """Keep the provider ID, retaining the fallback for older responses."""
    converted = _convert_response_to_response_dict(_response(call_id))
    call = converted["choices"][0]["message"]["tool_calls"][0]
    assert call["id"] == (call_id or f"call_{hash('get_weather')}_0")


@pytest.mark.parametrize("call_id", ["call_120847", None, ""])
def test_streaming_tool_call_id(call_id: str | None) -> None:
    """Provider IDs do not change the stream's monotonically increasing indices."""
    counter = [2]
    chunk = _create_openai_chunk_from_google_chunk(_response(call_id), counter)
    calls = chunk.choices[0].delta.tool_calls
    assert calls is not None
    assert calls[0].id == (call_id or f"call_{hash('get_weather')}_2")
    assert calls[0].index == 2
    assert counter == [3]


@pytest.mark.parametrize("stream", [False, True])
def test_parallel_tool_ids_round_trip(stream: bool) -> None:
    """Parallel calls to the same function retain IDs and signed parts on replay."""
    response = _response("call_120847")
    second_response = _response("call_120848", "Zurich")
    assert response.candidates
    assert response.candidates[0].content
    assert second_response.candidates
    assert second_response.candidates[0].content
    response.candidates[0].content.parts = (response.candidates[0].content.parts or []) + (
        second_response.candidates[0].content.parts or []
    )
    if stream:
        chunk = _create_openai_chunk_from_google_chunk(response)
        calls = chunk.choices[0].delta.tool_calls
        assert calls is not None
        tool_calls = [call.model_dump(exclude={"index"}) for call in calls]
    else:
        tool_calls = _convert_response_to_response_dict(response)["choices"][0]["message"]["tool_calls"]
    assert [call["id"] for call in tool_calls] == ["call_120847", "call_120848"]

    contents, _ = _convert_messages(
        [
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": "call_120848", "content": json.dumps({"temperature": 10})},
            {"role": "tool", "tool_call_id": "call_120847", "content": json.dumps({"temperature": 20})},
        ],
        "gemini",
    )
    assert contents[0].parts
    assert contents[1].parts
    function_calls = [part.function_call for part in contents[0].parts]
    assert all(call is not None for call in function_calls)
    assert [call.id for call in function_calls if call is not None] == ["call_120847", "call_120848"]
    assert [part.thought_signature for part in contents[0].parts] == [b"signature", b"signature"]
    results = [part.function_response for part in contents[1].parts]
    assert [result.id for result in results if result is not None] == ["call_120848", "call_120847"]
    assert [result.name for result in results if result is not None] == ["get_weather", "get_weather"]


@pytest.mark.parametrize("call_id", [None, ""])
def test_tool_history_without_ids(call_id: str | None) -> None:
    """Calls and results without IDs, or with empty ones, are replayed without an ID."""
    tool_call: dict[str, Any] = {"function": {"name": "get_weather", "arguments": "{}"}}
    tool_result: dict[str, Any] = {"role": "tool", "name": "get_weather", "content": "sunny"}
    if call_id is not None:
        tool_call["id"] = call_id
        tool_result["tool_call_id"] = call_id
    contents, _ = _convert_messages([{"role": "assistant", "tool_calls": [tool_call]}, tool_result], "gemini")
    assert contents[0].parts
    assert contents[0].parts[0].function_call
    assert contents[1].parts
    assert contents[1].parts[0].function_response
    assert contents[0].parts[0].function_call.id is None
    assert contents[1].parts[0].function_response.id is None


@pytest.mark.asyncio
async def test_tool_call_id_reaches_sdk_wire() -> None:
    """The official SDK receives matching IDs on the replayed call and result."""
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(200, json=_response("call_120847").model_dump(mode="json", by_alias=True))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "sunny"}]}}]},
        )

    provider = GeminiProvider(
        api_key="test-api-key",
        http_options=types.HttpOptions(async_client_args={"transport": httpx.MockTransport(handler)}),
    )
    try:
        result = await provider.acompletion(model="gemini-test", messages=[{"role": "user", "content": "Weather?"}])
        assert isinstance(result, ChatCompletion)
        message = result.choices[0].message
        assert message.tool_calls
        assert message.tool_calls[0].id == "call_120847"
        await provider.acompletion(
            model="gemini-test",
            messages=[message, {"role": "tool", "tool_call_id": "call_120847", "content": "sunny"}],
        )
    finally:
        await provider.client.aio.aclose()
    call = requests[1]["contents"][0]["parts"][0]["functionCall"]
    response = requests[1]["contents"][1]["parts"][0]["functionResponse"]
    assert call["id"] == response["id"] == "call_120847"
    assert call["name"] == response["name"] == "get_weather"
