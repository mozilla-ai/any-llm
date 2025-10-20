from typing import Any

import pytest

from any_llm.constants import LLMProvider
from tests.constants import INCLUDE_LOCAL_PROVIDERS, INCLUDE_NON_LOCAL_PROVIDERS, LOCAL_PROVIDERS


@pytest.fixture
def provider_reasoning_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
        LLMProvider.MISTRAL: "magistral-small-latest",
        LLMProvider.GEMINI: "gemini-2.5-flash",
        LLMProvider.VERTEXAI: "gemini-2.5-flash",
        LLMProvider.GROQ: "openai/gpt-oss-20b",
        LLMProvider.FIREWORKS: "accounts/fireworks/models/gpt-oss-20b",
        LLMProvider.OPENAI: "gpt-5-nano",
        LLMProvider.MISTRAL: "magistral-small-latest",
        LLMProvider.XAI: "grok-3-mini-latest",
        LLMProvider.OLLAMA: "qwen3:0.6b",
        LLMProvider.OPENROUTER: "deepseek/deepseek-v3.1-terminus",
        LLMProvider.LLAMAFILE: "N/A",
        LLMProvider.LLAMACPP: "N/A",
        LLMProvider.LMSTUDIO: "openai/gpt-oss-20b",  # You must have LM Studio running and the server enabled
        LLMProvider.AZUREOPENAI: "azure/<your_deployment_name>",
        LLMProvider.CEREBRAS: "gpt-oss-120b",
        LLMProvider.COHERE: "command-a-reasoning-08-2025",
        LLMProvider.DEEPSEEK: "deepseek-reasoner",
        LLMProvider.MOONSHOT: "kimi-thinking-preview",
        LLMProvider.DATABRICKS: "databricks-gpt-oss-20b",  # Untested, needs to be verified once we get a Databricks account
        LLMProvider.BEDROCK: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        LLMProvider.HUGGINGFACE: "huggingface/tgi",
        LLMProvider.NEBIUS: "openai/gpt-oss-20b",
        LLMProvider.SAMBANOVA: "DeepSeek-R1-Distill-Llama-70B",
        LLMProvider.TOGETHER: "OpenAI/gpt-oss-20B",
        LLMProvider.PORTKEY: "@anthropic/claude-3-7-sonnet-latest",
    }


# Use small models for testing to make sure they work
@pytest.fixture
def provider_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.MISTRAL: "mistral-small-latest",
        LLMProvider.ANTHROPIC: "claude-3-5-haiku-latest",
        LLMProvider.DATABRICKS: "databricks-meta-llama-3-1-8b-instruct",  # Untested, needs to be verified once we get a Databricks account
        LLMProvider.DEEPSEEK: "deepseek-chat",
        LLMProvider.OPENAI: "gpt-5-nano",
        LLMProvider.GEMINI: "gemini-2.5-flash",
        LLMProvider.VERTEXAI: "gemini-2.5-flash",
        LLMProvider.MOONSHOT: "moonshot-v1-8k",
        LLMProvider.SAMBANOVA: "Meta-Llama-3.1-8B-Instruct",
        LLMProvider.TOGETHER: "OpenAI/gpt-oss-20B",
        LLMProvider.XAI: "grok-3-mini-latest",
        LLMProvider.INCEPTION: "mercury",
        LLMProvider.NEBIUS: "openai/gpt-oss-20b",
        LLMProvider.OLLAMA: "llama3.2:1b",
        LLMProvider.LLAMAFILE: "N/A",
        LLMProvider.LMSTUDIO: "google/gemma-3n-e4b",  # You must have LM Studio running and the server enabled
        LLMProvider.COHERE: "command-a-03-2025",
        LLMProvider.CEREBRAS: "llama-3.3-70b",
        LLMProvider.HUGGINGFACE: "huggingface/tgi",  # This is the syntax used in `litellm` when using HF Inference Endpoints (https://docs.litellm.ai/docs/providers/huggingface#dedicated-inference-endpoints)
        LLMProvider.BEDROCK: "amazon.nova-lite-v1:0",
        LLMProvider.SAGEMAKER: "<sagemaker_endpoint_name>",
        LLMProvider.WATSONX: "ibm/granite-3-8b-instruct",
        LLMProvider.FIREWORKS: "accounts/fireworks/models/llama4-scout-instruct-basic",
        LLMProvider.GROQ: "openai/gpt-oss-20b",
        LLMProvider.PORTKEY: "@any-llm-test/gpt-4.1-nano",
        LLMProvider.LLAMA: "Llama-4-Maverick-17B-128E-Instruct-FP8",
        LLMProvider.AZURE: "openai/gpt-4.1-nano",
        LLMProvider.AZUREOPENAI: "azure/<your_deployment_name>",
        LLMProvider.PERPLEXITY: "sonar",
        LLMProvider.OPENROUTER: "meta-llama/llama-3.3-8b-instruct:free",
        LLMProvider.LLAMACPP: "N/A",
    }


@pytest.fixture
def provider_image_model_map(provider_model_map: dict[LLMProvider, str]) -> dict[LLMProvider, str]:
    return {
        **provider_model_map,
        LLMProvider.WATSONX: "meta-llama/llama-guard-3-11b-vision",
        LLMProvider.SAMBANOVA: "Llama-4-Maverick-17B-128E-Instruct",
        LLMProvider.NEBIUS: "openai/gpt-oss-20b",
        LLMProvider.OPENROUTER: "mistralai/mistral-small-3.2-24b-instruct:free",
        LLMProvider.OLLAMA: "llava:7b",
    }


# Embedding model map - only for providers that support embeddings
@pytest.fixture
def embedding_provider_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.OPENAI: "text-embedding-ada-002",
        LLMProvider.DATABRICKS: "databricks-bge-large-en",
        LLMProvider.NEBIUS: "Qwen/Qwen3-Embedding-8B",
        LLMProvider.SAMBANOVA: "Meta-Llama-3.1-8B-Instruct",
        LLMProvider.MISTRAL: "mistral-embed",
        LLMProvider.BEDROCK: "amazon.titan-embed-text-v2:0",
        LLMProvider.SAGEMAKER: "<sagemaker_endpoint_name>",
        LLMProvider.OLLAMA: "gpt-oss:20b",
        LLMProvider.LLAMAFILE: "N/A",
        LLMProvider.LMSTUDIO: "text-embedding-nomic-embed-text-v1.5",
        LLMProvider.GEMINI: "gemini-embedding-001",
        LLMProvider.VERTEXAI: "gemini-embedding-001",
        LLMProvider.AZURE: "openai/text-embedding-3-small",
        LLMProvider.AZUREOPENAI: "azure/<your_deployment_name>",
        LLMProvider.VOYAGE: "voyage-3.5-lite",
        LLMProvider.LLAMACPP: "N/A",
    }


@pytest.fixture
def provider_client_config() -> dict[LLMProvider, dict[str, Any]]:
    return {
        LLMProvider.ANTHROPIC: {"timeout": 10},
        LLMProvider.AZURE: {
            "api_base": "https://models.github.ai/inference",
        },
        LLMProvider.BEDROCK: {"region_name": "us-east-1"},
        LLMProvider.CEREBRAS: {"timeout": 10},
        LLMProvider.COHERE: {"timeout": 10},
        LLMProvider.DATABRICKS: {"api_base": "https://dbc-ec667410-1149.cloud.databricks.com/serving-endpoints"},
        LLMProvider.GROQ: {"timeout": 10},
        LLMProvider.HUGGINGFACE: {"api_base": "https://y0okp71n85ezo5nr.us-east-1.aws.endpoints.huggingface.cloud/v1/"},
        LLMProvider.LLAMACPP: {"api_base": "http://127.0.0.1:8090/v1"},
        LLMProvider.MISTRAL: {"timeout_ms": 100000},
        LLMProvider.NEBIUS: {"api_base": "https://api.studio.nebius.com/v1/"},
        LLMProvider.OPENAI: {"timeout": 10},
        LLMProvider.TOGETHER: {"timeout": 10},
        LLMProvider.VOYAGE: {"timeout": 10},
        LLMProvider.WATSONX: {
            "api_base": "https://us-south.ml.cloud.ibm.com",
            "project_id": "5b083ace-95a6-4f95-a0a0-d4c5d9e98ca0",
        },
        LLMProvider.XAI: {"timeout": 100},
    }


def _get_providers_for_testing() -> list[LLMProvider]:
    """Get the list of providers to test based on INCLUDE_LOCAL_PROVIDERS and INCLUDE_NON_LOCAL_PROVIDERS settings."""
    all_providers = list(LLMProvider)

    filtered = []
    if INCLUDE_LOCAL_PROVIDERS:
        filtered.extend([provider for provider in all_providers if provider in LOCAL_PROVIDERS])
    if INCLUDE_NON_LOCAL_PROVIDERS:
        filtered.extend([provider for provider in all_providers if provider not in LOCAL_PROVIDERS])

    return filtered


@pytest.fixture(params=_get_providers_for_testing(), ids=lambda x: x.value)
def provider(request: pytest.FixtureRequest) -> LLMProvider:
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
