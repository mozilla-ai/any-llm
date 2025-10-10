import pytest
from any_llm.providers.cohere import utils
from any_llm.types.completion import ChatCompletionChunk, ChatCompletion, ChatCompletionMessageFunctionToolCall, Function, Choice, ChatCompletionMessage
from any_llm.types.model import Model
from types import SimpleNamespace

# --- _patch_messages ---
def test_patch_messages_tool_removes_name():
    messages = [
        {"role": "assistant", "tool_calls": True, "content": "plan"},
        {"role": "tool", "name": "toolname", "content": "result"},
    ]
    patched = utils._patch_messages(messages)
    assert "name" not in patched[1]
    assert patched[1]["role"] == "tool"

def test_patch_messages_tool_wrong_sequence_raises():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "name": "toolname", "content": "result"},
    ]
    with pytest.raises(ValueError):
        utils._patch_messages(messages)

def test_patch_messages_assistant_tool_calls_content_to_tool_plan():
    messages = [
        {"role": "assistant", "tool_calls": True, "content": "plan"},
    ]
    patched = utils._patch_messages(messages)
    assert "tool_plan" in patched[0]
    assert "content" not in patched[0]

# --- _create_openai_chunk_from_cohere_chunk ---
class DummyDelta:
    def __init__(self, text=None, tool_calls=None, usage=None):
        self.message = SimpleNamespace(content=SimpleNamespace(text=text), tool_calls=tool_calls)
        self.usage = usage

class DummyChunk:
    def __init__(self, type, delta=None):
        self.type = type
        self.delta = delta

class DummyToolCall:
    def __init__(self, id="id", function=None):
        self.id = id
        self.function = function

class DummyFunction:
    def __init__(self, name="fname", arguments="args"):
        self.name = name
        self.arguments = arguments

class DummyUsage:
    def __init__(self, input_tokens=1, output_tokens=2):
        self.tokens = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)

@pytest.mark.parametrize("chunk_type,text,expected", [
    ("content-delta", "hello", "hello"),
    ("tool-call-start", None, None),
    ("tool-call-delta", None, None),
    ("tool-call-end", None, None),
    ("message-end", None, None),
])
def test_create_openai_chunk_from_cohere_chunk(chunk_type, text, expected):
    if chunk_type == "content-delta":
        delta = DummyDelta(text=text)
    elif chunk_type == "tool-call-start":
        delta = DummyDelta(tool_calls=DummyToolCall(function=DummyFunction()))
    elif chunk_type == "tool-call-delta":
        tc = DummyToolCall(function=DummyFunction())
        delta = DummyDelta(tool_calls=tc)
    elif chunk_type == "tool-call-end":
        delta = DummyDelta()
    elif chunk_type == "message-end":
        usage = DummyUsage()
        delta = DummyDelta(usage=usage)
    chunk = DummyChunk(type=chunk_type, delta=delta)
    result = utils._create_openai_chunk_from_cohere_chunk(chunk)
    assert isinstance(result, ChatCompletionChunk)

# --- _convert_response ---
class DummyResponse:
    def __init__(self, finish_reason=None, message=None, usage=None, id="id", created=0):
        self.finish_reason = finish_reason
        self.message = message
        self.usage = usage
        self.id = id
        self.created = created

class DummyMessage:
    def __init__(self, tool_calls=None, tool_plan=None, content=None):
        self.tool_calls = tool_calls
        self.tool_plan = tool_plan
        self.content = content

class DummyToolCallObj:
    def __init__(self, id="tid", function=None):
        self.id = id
        self.function = function

class DummyFunctionObj:
    def __init__(self, name="fname", arguments="args"):
        self.name = name
        self.arguments = arguments

class DummyUsageObj:
    def __init__(self, input_tokens=1, output_tokens=2):
        self.tokens = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)

def test_convert_response_tool_call():
    tool_call = DummyToolCallObj(function=DummyFunctionObj())
    message = DummyMessage(tool_calls=[tool_call], tool_plan="plan", content=None)
    usage = DummyUsageObj()
    response = DummyResponse(finish_reason="TOOL_CALL", message=message, usage=usage)
    result = utils._convert_response(response, "cohere-model")
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls[0].function.name == "fname"

def test_convert_response_content():
    message = DummyMessage(tool_calls=None, tool_plan=None, content=[SimpleNamespace(text="hello")])
    usage = DummyUsageObj()
    response = DummyResponse(finish_reason="stop", message=message, usage=usage)
    result = utils._convert_response(response, "cohere-model")
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "hello"

# --- _convert_models_list ---
class DummyModelData:
    def __init__(self, name):
        self.name = name
class DummyListModelsResponse:
    def __init__(self, models):
        self.models = models

def test_convert_models_list():
    models = [DummyModelData("m1"), DummyModelData("m2")]
    response = DummyListModelsResponse(models)
    result = utils._convert_models_list(response)
    assert isinstance(result, list)
    assert all(isinstance(m, Model) for m in result)
    assert result[0].id == "m1"
    assert result[1].id == "m2"

def test_convert_models_list_empty():
    response = DummyListModelsResponse(models=None)
    result = utils._convert_models_list(response)
    assert result == []
