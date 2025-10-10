import pytest
from any_llm.providers.anthropic import utils
from types import SimpleNamespace

# Test _is_tool_call
def test_is_tool_call_true():
    msg = {"role": "assistant", "tool_calls": [{}]}
    assert utils._is_tool_call(msg)
def test_is_tool_call_false():
    msg = {"role": "user"}
    assert not utils._is_tool_call(msg)
# Test _convert_messages_for_anthropic
def test_convert_messages_for_anthropic():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "result"},
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}}]}
    ]
    sys_msg, filtered = utils._convert_messages_for_anthropic(messages)
    assert sys_msg == "sys"
    assert isinstance(filtered, list)
# Test _create_openai_chunk_from_anthropic_chunk
def test_create_openai_chunk_from_anthropic_chunk():
    class DummyChunk:
        def __str__(self): return "chunk"
    result = utils._create_openai_chunk_from_anthropic_chunk(DummyChunk(), "test-model")
    assert hasattr(result, "choices")
# Test _convert_response
def test_convert_response():
    class DummyContent:
        def __init__(self, type, text=None, id=None, name=None, input=None, thinking=None):
            self.type = type
            self.text = text
            self.id = id
            self.name = name
            self.input = input
            self.thinking = thinking
    class DummyUsage:
        input_tokens = 1
        output_tokens = 2
    class DummyResponse:
        stop_reason = "end_turn"
        content = [DummyContent("text", text="hi")]
        usage = DummyUsage()
        id = "id1"
        model = "test-model"
        created_at = SimpleNamespace(timestamp=lambda: 123)
    result = utils._convert_response(DummyResponse())
    assert hasattr(result, "choices")
# Test _convert_tool_spec
def test_convert_tool_spec():
    openai_tools = [
        {"type": "function", "function": {"name": "f", "description": "desc", "parameters": {"properties": {"p": {}}, "required": ["p"]}}}
    ]
    result = utils._convert_tool_spec(openai_tools)
    assert isinstance(result, list)
    assert result[0]["name"] == "f"
# Test _convert_tool_choice
def test_convert_tool_choice():
    class DummyParams:
        parallel_tool_calls = None
        tool_choice = "required"
    result = utils._convert_tool_choice(DummyParams())
    assert result["type"] == "any"
# Test _convert_params
def test_convert_params():
    class DummyParams:
        response_format = None
        max_tokens = None
        tools = None
        tool_choice = None
        parallel_tool_calls = None
        reasoning_effort = None
        model_id = "test-model"
        messages = [{"role": "user", "content": "hi"}]
        def model_dump(self, **kwargs): return {"max_tokens": 10}
    result = utils._convert_params(DummyParams(), provider_name="anthropic")
    assert result["model"] == "test-model"
# Test _convert_models_list
def test_convert_models_list():
    class DummyModel:
        def __init__(self, id): self.id = id; self.created_at = SimpleNamespace(timestamp=lambda: 123)
    models = [DummyModel("m1"), DummyModel("m2")]
    result = utils._convert_models_list(models)
    assert result[0].id == "m1"
