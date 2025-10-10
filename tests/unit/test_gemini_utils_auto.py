import pytest
from any_llm.providers.gemini import utils
from types import SimpleNamespace

# Mocks for google.generativeai.types
class MockFunctionDeclaration:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters
class MockSchema:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
class MockTool:
    def __init__(self, function_declarations):
        self.function_declarations = function_declarations
class MockPart:
    @staticmethod
    def from_text(text):
        return {"type": "text", "text": text}
    @staticmethod
    def from_bytes(data, mime_type):
        return {"type": "bytes", "data": data, "mime_type": mime_type}
    @staticmethod
    def from_function_call(name, args):
        return {"type": "function_call", "name": name, "args": args}
    @staticmethod
    def from_function_response(name, response):
        return {"type": "function_response", "name": name, "response": response}
class MockContent:
    def __init__(self, role, parts):
        self.role = role
        self.parts = parts
class MockEmbed:
    def __init__(self, values):
        self.values = values
class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.embeddings = embeddings
class MockGenerateContentResponse:
    def __init__(self, candidates, usage_metadata=None, model_version="test-model"):
        self.candidates = candidates
        self.usage_metadata = usage_metadata or SimpleNamespace(prompt_token_count=1, candidates_token_count=2, total_token_count=3)
        self.model_version = model_version
class MockCandidate:
    def __init__(self, content, finish_reason=None):
        self.content = content
        self.finish_reason = finish_reason or SimpleNamespace(value="STOP")
class MockModel:
    def __init__(self, name):
        self.name = name
# Patch types in utils
utils.types.FunctionDeclaration = MockFunctionDeclaration
utils.types.Schema = MockSchema
utils.types.Tool = MockTool
utils.types.Part = MockPart
utils.types.Content = MockContent
utils.types.EmbedContentResponse = MockEmbeddingResponse
utils.types.GenerateContentResponse = MockGenerateContentResponse
utils.types.Model = MockModel
# Test _convert_tool_spec
def test_convert_tool_spec():
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "test_func",
                "description": "desc",
                "parameters": {
                    "properties": {
                        "param1": {"type": "string", "description": "desc1"}
                    },
                    "required": ["param1"]
                }
            }
        }
    ]
    result = utils._convert_tool_spec(openai_tools)
    assert isinstance(result, list)
    assert isinstance(result[0], MockTool)
# Test _convert_messages
def test_convert_messages():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "{}", "name": "tool1"}
    ]
    result, sys_inst = utils._convert_messages(messages)
    assert isinstance(result, list)
    assert sys_inst == "sys"
# Test _convert_response_to_response_dict
def test_convert_response_to_response_dict():
    part = SimpleNamespace(text="response", thought=None)
    candidate = MockCandidate(content=SimpleNamespace(parts=[part]))
    response = MockGenerateContentResponse([candidate])
    result = utils._convert_response_to_response_dict(response)
    assert "choices" in result
# Test _create_openai_embedding_response_from_google
def test_create_openai_embedding_response_from_google():
    embed1 = MockEmbed([0.1, 0.2, 0.3])
    response = MockEmbeddingResponse([embed1])
    result = utils._create_openai_embedding_response_from_google("test-model", response)
    assert hasattr(result, "data")
# Test _create_openai_chunk_from_google_chunk
def test_create_openai_chunk_from_google_chunk():
    part = SimpleNamespace(text="chunk", thought=None)
    candidate = MockCandidate(content=SimpleNamespace(parts=[part]))
    response = MockGenerateContentResponse([candidate])
    result = utils._create_openai_chunk_from_google_chunk(response)
    assert hasattr(result, "choices")
# Test _convert_models_list
def test_convert_models_list():
    models = [MockModel("model1"), MockModel("model2")]
    result = utils._convert_models_list(models)
    assert isinstance(result, list)
    assert result[0].id == "model1"
