from typing import Any

import pytest

from any_llm.provider import ProviderName


# Use small models for testing to make sure they work
@pytest.fixture
def provider_model_map() -> dict[ProviderName, str]:
    return {
        ProviderName.MISTRAL: "mistral-small-latest",
        ProviderName.ANTHROPIC: "claude-3-5-haiku-latest",
        ProviderName.DEEPSEEK: "deepseek-chat",
        ProviderName.OPENAI: "gpt-4.1-mini",
        ProviderName.GOOGLE: "gemini-2.0-flash-001",
        ProviderName.MOONSHOT: "moonshot-v1-8k",
        ProviderName.SAMBANOVA: "Meta-Llama-3.1-8B-Instruct",
        ProviderName.TOGETHER: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        ProviderName.XAI: "grok-3-mini-latest",
        ProviderName.INCEPTION: "inception-3-70b-instruct",
        ProviderName.NEBIUS: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ProviderName.OLLAMA: "llama3.2:1b",
        ProviderName.AZURE: "gpt-4o",
        ProviderName.COHERE: "command-a-03-2025",
        ProviderName.CEREBRAS: "llama-3.3-70b",
        ProviderName.HUGGINGFACE: "meta-llama/Llama-3.2-3B-Instruct",  # You must have novita enabled in your hf account to use this model
        ProviderName.AWS: "amazon.nova-lite-v1:0",
        ProviderName.WATSONX: "google/gemini-2.0-flash-001",
        ProviderName.FIREWORKS: "accounts/fireworks/models/llama4-scout-instruct-basic",
        ProviderName.GROQ: "llama-3.1-8b-instant",
    }


# Embedding model map - only for providers that support embeddings
@pytest.fixture
def embedding_provider_model_map() -> dict[ProviderName, str]:
    return {
        ProviderName.OPENAI: "text-embedding-ada-002",
        ProviderName.NEBIUS: "Qwen/Qwen3-Embedding-8B",
        ProviderName.SAMBANOVA: "E5-Mistral-7B-Instruct",
        ProviderName.MISTRAL: "mistral-embed",
        ProviderName.AWS: "amazon.titan-embed-text-v2:0",
        ProviderName.OLLAMA: "llama3.2:1b",
    }


@pytest.fixture(params=list(ProviderName), ids=lambda x: x.value)
def provider(request: pytest.FixtureRequest) -> ProviderName:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture()
def tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and country e.g. Paris, France"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]
