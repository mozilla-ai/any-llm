import pytest
from any_llm.providers.gemini.base import GoogleProvider
from any_llm.types.completion import CompletionParams, ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse, Choice, Function, Reasoning
from any_llm.types.model import Model
from types import SimpleNamespace

class DummyProvider(GoogleProvider):
    def _get_client(self, config):
        return None

def test_convert_completion_params():
    params = CompletionParams(model_id="test-model", max_tokens=10)
    result = GoogleProvider._convert_completion_params(params)
    assert result["max_output_tokens"] == 10

def test_convert_completion_response():
    response_dict = {
        "choices": [
            {"message": {"content": "hi", "reasoning": "r", "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": "{}"}, "type": "function"}]}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "id": "id1", "model": "test-model", "created": 123
    }
    result = GoogleProvider._convert_completion_response((response_dict, "test-model"))
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content == "hi"

def test_convert_completion_chunk_response():
    class DummyResponse: pass
    result = GoogleProvider._convert_completion_chunk_response(DummyResponse())
    assert isinstance(result, ChatCompletionChunk) or result is not None

def test_convert_embedding_params():
    result = GoogleProvider._convert_embedding_params(["abc"])
    assert result["contents"] == ["abc"]

def test_convert_embedding_response():
    class DummyEmbedding:
        def __init__(self):
            self.values = [0.1, 0.2]
    response = {"model": "test-model", "result": SimpleNamespace(embeddings=[DummyEmbedding()])}
    result = GoogleProvider._convert_embedding_response(response)
    assert isinstance(result, CreateEmbeddingResponse)

def test_convert_list_models_response():
    models = [SimpleNamespace(name="model1"), SimpleNamespace(name="model2")]
    result = GoogleProvider._convert_list_models_response(models)
    assert isinstance(result, list)
    assert result[0].id == "model1"
