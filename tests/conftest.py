import os
from collections import defaultdict
from collections.abc import Generator
from typing import Any

import pytest

from any_llm.constants import LLMProvider
from tests.constants import CI_EXCLUDED_PROVIDERS, INCLUDE_LOCAL_PROVIDERS, INCLUDE_NON_LOCAL_PROVIDERS, LOCAL_PROVIDERS


@pytest.fixture
def provider_reasoning_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.ANTHROPIC: "claude-sonnet-4-0",
        LLMProvider.MISTRAL: "magistral-small-latest",
        LLMProvider.GEMINI: "gemini-2.5-flash",
        LLMProvider.GATEWAY: "gpt-5-nano",
        LLMProvider.VERTEXAI: "gemini-2.5-flash",
        LLMProvider.GROQ: "openai/gpt-oss-20b",
        LLMProvider.FIREWORKS: "accounts/fireworks/models/gpt-oss-20b",
        LLMProvider.OPENAI: "gpt-5-nano",
        LLMProvider.MISTRAL: "magistral-medium-latest",
        LLMProvider.XAI: "grok-3-mini-latest",
        LLMProvider.OLLAMA: "qwen3:0.6b",
        LLMProvider.OPENROUTER: "google/gemini-2.5-flash-lite",
        LLMProvider.LLAMAFILE: "N/A",
        LLMProvider.LLAMACPP: "N/A",
        LLMProvider.VLLM: "N/A",
        LLMProvider.LMSTUDIO: "openai/gpt-oss-20b",  # You must have LM Studio running and the server enabled
        LLMProvider.AZUREOPENAI: "gpt-4.1-nano",
        LLMProvider.CEREBRAS: "gpt-oss-120b",
        LLMProvider.COHERE: "command-a-reasoning-08-2025",
        LLMProvider.DEEPSEEK: "deepseek-reasoner",
        LLMProvider.MOONSHOT: "kimi-k2-thinking",
        LLMProvider.BEDROCK: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        LLMProvider.HUGGINGFACE: "huggingface/tgi",
        LLMProvider.NEBIUS: "openai/gpt-oss-20b",
        LLMProvider.SAMBANOVA: "DeepSeek-R1-Distill-Llama-70B",
        LLMProvider.TOGETHER: "openai/gpt-oss-20b",
        LLMProvider.PORTKEY: "@nebius-any-llm/Qwen/Qwen3-32B",
        LLMProvider.MINIMAX: "MiniMax-M2",
        LLMProvider.ZAI: "glm-4.5-flash",
    }


# Use small models for testing to make sure they work
@pytest.fixture
def provider_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.MISTRAL: "mistral-small-latest",
        LLMProvider.ANTHROPIC: "claude-haiku-4-5",
        LLMProvider.DEEPSEEK: "deepseek-chat",
        LLMProvider.OPENAI: "gpt-5-nano",
        LLMProvider.GATEWAY: "gpt-5-nano",
        LLMProvider.GEMINI: "gemini-3-flash-preview",
        LLMProvider.VERTEXAI: "gemini-3-flash-preview",
        LLMProvider.MOONSHOT: "moonshot-v1-8k",
        LLMProvider.SAMBANOVA: "gpt-oss-120b",
        LLMProvider.TOGETHER: "openai/gpt-oss-20b",
        LLMProvider.XAI: "grok-3-mini-latest",
        LLMProvider.INCEPTION: "mercury",
        LLMProvider.NEBIUS: "openai/gpt-oss-20b",
        LLMProvider.OLLAMA: "llama3.2:1b",
        LLMProvider.LLAMAFILE: "N/A",
        LLMProvider.LMSTUDIO: "google/gemma-3n-e4b",  # You must have LM Studio running and the server enabled
        LLMProvider.VLLM: "Qwen/Qwen2.5-0.5B-Instruct",
        LLMProvider.COHERE: "command-a-03-2025",
        LLMProvider.CEREBRAS: "llama3.1-8b",
        LLMProvider.HUGGINGFACE: "huggingface/tgi",  # This is the syntax used in `litellm` when using HF Inference Endpoints (https://docs.litellm.ai/docs/providers/huggingface#dedicated-inference-endpoints)
        LLMProvider.BEDROCK: "amazon.nova-lite-v1:0",
        LLMProvider.SAGEMAKER: "<sagemaker_endpoint_name>",
        LLMProvider.WATSONX: "ibm/granite-3-8b-instruct",
        LLMProvider.FIREWORKS: "accounts/fireworks/models/gpt-oss-20b",
        LLMProvider.GROQ: "openai/gpt-oss-20b",
        LLMProvider.PORTKEY: "@any-llm-test/gpt-4.1-mini",
        LLMProvider.LLAMA: "Llama-4-Maverick-17B-128E-Instruct-FP8",
        LLMProvider.AZURE: "openai/gpt-4.1-nano",
        LLMProvider.AZUREOPENAI: "gpt-4.1-nano",
        LLMProvider.PERPLEXITY: "sonar",
        LLMProvider.OPENROUTER: "google/gemini-2.5-flash-lite",
        LLMProvider.LLAMACPP: "N/A",
        LLMProvider.MINIMAX: "MiniMax-M2",
        LLMProvider.ZAI: "glm-4-32b-0414-128k",
    }


@pytest.fixture
def provider_image_model_map(provider_model_map: dict[LLMProvider, str]) -> dict[LLMProvider, str]:
    return {
        **provider_model_map,
        LLMProvider.OPENAI: "gpt-5-mini",  # Slightly more powerful so that it doesn't get caught in a loop of logic
        LLMProvider.WATSONX: "meta-llama/llama-guard-3-11b-vision",
        LLMProvider.SAMBANOVA: "Llama-4-Maverick-17B-128E-Instruct",
        LLMProvider.NEBIUS: "openai/gpt-oss-20b",
        LLMProvider.OPENROUTER: "google/gemini-2.5-flash-lite",
        LLMProvider.OLLAMA: "llava-phi3",  # Fast vision model compatible with OpenAI format
        LLMProvider.FIREWORKS: "accounts/fireworks/models/kimi-k2p5",
        LLMProvider.BEDROCK: "anthropic.claude-3-haiku-20240307-v1:0",  # Claude 3 Haiku with vision support
    }


# Embedding model map - only for providers that support embeddings
@pytest.fixture
def embedding_provider_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.OPENAI: "text-embedding-ada-002",
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
        LLMProvider.VOYAGE: "voyage-3.5-lite",
        LLMProvider.LLAMACPP: "N/A",
        LLMProvider.GATEWAY: "text-embedding-ada-002",
        LLMProvider.AZUREOPENAI: "gpt-4.1-nano",  # Not an embedding model but it's the only one we have deployed in Azure OpenAI
        LLMProvider.OPENROUTER: "qwen/qwen3-embedding-8b",
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
        LLMProvider.GATEWAY: {"api_base": "http://127.0.0.1:3000", "timeout": 1},
        LLMProvider.GROQ: {"timeout": 10},
        LLMProvider.OPENAI: {"timeout": 100},
        LLMProvider.HUGGINGFACE: {"api_base": "https://oze7k8n86bjfzgjk.us-east-1.aws.endpoints.huggingface.cloud/v1"},
        LLMProvider.LLAMACPP: {"api_base": "http://127.0.0.1:8090/v1"},
        LLMProvider.VLLM: {"api_base": "http://127.0.0.1:8080/v1"},
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
        LLMProvider.AZUREOPENAI: {
            "api_base": "https://mlrun-me8bof5t-eastus2.cognitiveservices.azure.com/",
            "api_version": "2025-03-01-preview",
        },
    }


def _get_providers_for_testing() -> list[LLMProvider]:
    """Get the list of providers to test based on INCLUDE_LOCAL_PROVIDERS and INCLUDE_NON_LOCAL_PROVIDERS settings."""
    all_providers = list(LLMProvider)

    filtered = []
    if INCLUDE_LOCAL_PROVIDERS:
        filtered.extend([provider for provider in all_providers if provider in LOCAL_PROVIDERS])
    if INCLUDE_NON_LOCAL_PROVIDERS:
        filtered.extend([provider for provider in all_providers if provider not in LOCAL_PROVIDERS])

    return [provider for provider in filtered if provider not in CI_EXCLUDED_PROVIDERS]


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


# =============================================================================
# Retry Statistics Tracking (for pytest-rerunfailures integration)
# =============================================================================

_retry_stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"attempts": 1, "final_outcome": "passed"})


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Generator[None, Any, None]:
    """Track retry attempts for each test using pytest-rerunfailures execution_count."""
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call":
        test_name = item.nodeid
        execution_count = getattr(item, "execution_count", 1)

        _retry_stats[test_name]["attempts"] = execution_count
        _retry_stats[test_name]["final_outcome"] = rep.outcome


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Add retry summary to terminal output."""
    retried_tests = {k: v for k, v in _retry_stats.items() if v["attempts"] > 1}

    if not retried_tests:
        return

    terminalreporter.write_sep("=", "RETRY SUMMARY")
    terminalreporter.write_line(f"Tests that required retries: {len(retried_tests)}")
    terminalreporter.write_line("")

    for test_name, stats in sorted(retried_tests.items()):
        terminalreporter.write_line(f"  {test_name}: {stats['attempts']} attempts, final: {stats['final_outcome']}")


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write retry summary to GitHub Actions job summary (GITHUB_STEP_SUMMARY)."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    retried_tests = {k: v for k, v in _retry_stats.items() if v["attempts"] > 1}

    if not retried_tests:
        return

    with open(summary_file, "a") as f:
        f.write("\n## 🔄 Test Retry Summary\n\n")
        f.write(f"**{len(retried_tests)} test(s) required retries**\n\n")
        f.write("| Test | Attempts | Final Result |\n")
        f.write("|------|----------|-------------|\n")

        for test_name, stats in sorted(retried_tests.items()):
            short_name = test_name.split("::")[-1] if "::" in test_name else test_name
            result_emoji = "✅" if stats["final_outcome"] == "passed" else "❌"
            f.write(f"| `{short_name}` | {stats['attempts']} | {result_emoji} {stats['final_outcome']} |\n")
