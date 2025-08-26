from typing import Any

import pytest

from any_llm.provider import ProviderName


@pytest.fixture
def provider_reasoning_model_map() -> dict[ProviderName, str]:
    return {
        ProviderName.ANTHROPIC: "claude-sonnet-4-20250514",
        ProviderName.MISTRAL: "magistral-small-latest",
        ProviderName.GOOGLE: "gemini-2.5-flash",
        ProviderName.GROQ: "openai/gpt-oss-20b",
        ProviderName.FIREWORKS: "accounts/fireworks/models/deepseek-r1",
        ProviderName.OPENAI: "gpt-5-nano",
        ProviderName.MISTRAL: "magistral-small-latest",
        ProviderName.XAI: "grok-3-mini-latest",
        ProviderName.OLLAMA: "qwen3:0.6b",
        ProviderName.LLAMAFILE: "N/A",
        ProviderName.LLAMACPP: "N/A",
        ProviderName.LMSTUDIO: "openai/gpt-oss-20b",  # You must have LM Studio running and the server enabled
        ProviderName.AZUREOPENAI: "azure/<your_deployment_name>",
    }


# Use small models for testing to make sure they work
@pytest.fixture
def provider_model_map() -> dict[ProviderName, str]:
    return {
        ProviderName.MISTRAL: "mistral-small-latest",
        ProviderName.ANTHROPIC: "claude-3-5-haiku-latest",
        ProviderName.DATABRICKS: "databricks-meta-llama-3-1-8b-instruct",
        ProviderName.DEEPSEEK: "deepseek-chat",
        ProviderName.OPENAI: "gpt-5-nano",
        ProviderName.GOOGLE: "gemini-2.5-flash",
        ProviderName.MOONSHOT: "moonshot-v1-8k",
        ProviderName.SAMBANOVA: "Meta-Llama-3.1-8B-Instruct",
        ProviderName.TOGETHER: "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        ProviderName.XAI: "grok-3-mini-latest",
        ProviderName.INCEPTION: "inception-3-70b-instruct",
        ProviderName.NEBIUS: "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ProviderName.OLLAMA: "llama3.2:1b",
        ProviderName.LLAMAFILE: "N/A",
        ProviderName.LMSTUDIO: "google/gemma-3n-e4b",  # You must have LM Studio running and the server enabled
        ProviderName.COHERE: "command-a-03-2025",
        ProviderName.CEREBRAS: "llama-3.3-70b",
        ProviderName.HUGGINGFACE: "huggingface/tgi",  # This is the syntax used in `litellm` when using HF Inference Endpoints (https://docs.litellm.ai/docs/providers/huggingface#dedicated-inference-endpoints)
        ProviderName.AWS: "amazon.nova-lite-v1:0",
        ProviderName.WATSONX: "ibm/granite-3-8b-instruct",
        ProviderName.FIREWORKS: "accounts/fireworks/models/llama4-scout-instruct-basic",
        ProviderName.GROQ: "llama-3.1-8b-instant",
        ProviderName.OPENROUTER: "moonshotai/kimi-k2:free",
        ProviderName.PORTKEY: "@first-integrati-d8a10f/gpt-4.1-mini",  # Owned by njbrake in portkey UI
        ProviderName.LLAMA: "Llama-4-Maverick-17B-128E-Instruct-FP8",
        ProviderName.AZURE: "openai/gpt-4.1-nano",
        ProviderName.AZUREOPENAI: "azure/<your_deployment_name>",
        ProviderName.PERPLEXITY: "llama-3.1-sonar-small-128k-chat",
        ProviderName.LLAMACPP: "N/A",
    }


# Embedding model map - only for providers that support embeddings
@pytest.fixture
def embedding_provider_model_map() -> dict[ProviderName, str]:
    return {
        ProviderName.OPENAI: "text-embedding-ada-002",
        ProviderName.DATABRICKS: "databricks-bge-large-en",
        ProviderName.NEBIUS: "Qwen/Qwen3-Embedding-8B",
        ProviderName.SAMBANOVA: "E5-Mistral-7B-Instruct",
        ProviderName.MISTRAL: "mistral-embed",
        ProviderName.AWS: "amazon.titan-embed-text-v2:0",
        ProviderName.OLLAMA: "gpt-oss:20b",
        ProviderName.LLAMAFILE: "N/A",
        ProviderName.LMSTUDIO: "text-embedding-nomic-embed-text-v1.5",
        ProviderName.GOOGLE: "gemini-embedding-001",
        ProviderName.AZURE: "openai/text-embedding-3-small",
        ProviderName.AZUREOPENAI: "azure/<your_deployment_name>",
        ProviderName.VOYAGE: "voyage-3.5-lite",
        ProviderName.LLAMACPP: "N/A",
    }


@pytest.fixture
def provider_extra_kwargs_map() -> dict[ProviderName, dict[str, Any]]:
    return {
        ProviderName.AZURE: {
            "api_base": "https://models.github.ai/inference",
        },
        ProviderName.DATABRICKS: {"api_base": "https://dbc-40d03128-ecae.cloud.databricks.com/serving-endpoints"},
        ProviderName.HUGGINGFACE: {
            "api_base": "https://y0okp71n85ezo5nr.us-east-1.aws.endpoints.huggingface.cloud/v1/"
        },
        ProviderName.WATSONX: {
            "api_base": "https://us-south.ml.cloud.ibm.com",
            "project_id": "5b083ace-95a6-4f95-a0a0-d4c5d9e98ca0",
        },
    }


@pytest.fixture(params=list(ProviderName), ids=lambda x: x.value)
def provider(request: pytest.FixtureRequest) -> ProviderName:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
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


@pytest.fixture
def agent_loop_messages() -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": "What is the weather like in Salvaterra?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "foo", "function": {"name": "get_weather", "arguments": '{"location": "Salvaterra"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "foo", "content": "sunny"},
    ]
