import pytest
from any_llm.providers.llama import LlamaProvider
from any_llm.exceptions import UnsupportedParameterError

def test_verify_kwargs():
    provider = LlamaProvider(config={"api_key": "test_key"})

    # Test unsupported stream parameter
    with pytest.raises(UnsupportedParameterError):
        provider.verify_kwargs({"stream": True})

def test_make_api_call():
    provider = LlamaProvider(config={"api_key": "test_key"})

    # Mock the LlamaClient and its response
    class MockLlamaClient:
        def chat_completion(self, model, messages, **kwargs):
            return {"choices": [{"message": "Test response"}]}

    provider._make_api_call = MockLlamaClient().chat_completion

    response = provider._make_api_call(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response["choices"][0]["message"] == "Test response"
