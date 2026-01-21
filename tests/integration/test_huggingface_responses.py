"""Integration tests for HuggingFace OpenResponses API.

These tests verify that the HuggingFace provider correctly integrates with
the OpenResponses router at https://router.huggingface.co/v1.

To run these tests, you need:
1. A valid HF_TOKEN environment variable
2. Network access to the HuggingFace router

Run with: pytest tests/integration/test_huggingface_responses.py -v
"""

from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.responses import Response


@pytest.fixture
def huggingface_responses_model() -> str:
    """Model to use for HuggingFace OpenResponses tests.

    Uses the model:provider format for routing through the HuggingFace router.
    """
    return "openai/gpt-oss-120b:groq"


@pytest.fixture
def huggingface_client_config() -> dict[str, Any]:
    """Client configuration for HuggingFace provider."""
    return {}


@pytest.mark.asyncio
async def test_huggingface_responses_basic(
    huggingface_responses_model: str,
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test basic HuggingFace OpenResponses API call."""
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        assert llm.SUPPORTS_RESPONSES is True, "HuggingFace should support OpenResponses"

        result = await llm.aresponses(
            model=huggingface_responses_model,
            input_data="What is 2 + 2? Answer with just the number.",
        )

        assert isinstance(result, Response)
        assert result.output_text is not None
        assert "4" in result.output_text

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")


@pytest.mark.asyncio
async def test_huggingface_responses_with_instructions(
    huggingface_responses_model: str,
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test HuggingFace OpenResponses with system instructions."""
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        result = await llm.aresponses(
            model=huggingface_responses_model,
            input_data="What is your purpose?",
            instructions="You are a helpful math tutor. Always respond with enthusiasm!",
        )

        assert isinstance(result, Response)
        assert result.output_text is not None

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")


@pytest.mark.asyncio
async def test_huggingface_responses_streaming(
    huggingface_responses_model: str,
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test HuggingFace OpenResponses with streaming enabled."""
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        result = await llm.aresponses(
            model=huggingface_responses_model,
            input_data="Count from 1 to 5.",
            stream=True,
        )

        # Verify we get an async iterator (not a Response)
        assert not isinstance(result, Response)

        events_received = 0
        async for event in result:
            events_received += 1
            # Each event should be a ResponseStreamEvent
            assert event is not None

        # Should have received at least some events
        assert events_received > 0

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")


@pytest.mark.asyncio
async def test_huggingface_responses_with_tools(
    huggingface_responses_model: str,
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test HuggingFace OpenResponses with tool calling."""
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]

        result = await llm.aresponses(
            model=huggingface_responses_model,
            input_data="What is the weather like in Paris?",
            tools=tools,
            tool_choice="auto",
        )

        assert isinstance(result, Response)
        # The model should either call the tool or respond directly
        assert result.output is not None

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")


@pytest.mark.asyncio
async def test_huggingface_responses_with_reasoning(
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test HuggingFace OpenResponses with reasoning effort parameter."""
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        # Use a reasoning-capable model
        result = await llm.aresponses(
            model="openai/gpt-oss-120b:groq",
            input_data="What is the square root of 144? Think step by step.",
            reasoning={"effort": "medium"},
        )

        assert isinstance(result, Response)
        assert result.output_text is not None
        # The answer should contain 12
        assert "12" in result.output_text

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")


@pytest.mark.asyncio
async def test_huggingface_responses_different_providers(
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test HuggingFace OpenResponses with different provider routing.

    The model:provider syntax allows routing to different inference providers.
    """
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        # Test with Groq as the provider
        result = await llm.aresponses(
            model="moonshotai/Kimi-K2-Instruct-0905:groq",
            input_data="Say hello in one word.",
        )

        assert isinstance(result, Response)
        assert result.output_text is not None

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")


@pytest.mark.asyncio
async def test_huggingface_responses_multimodal(
    huggingface_client_config: dict[str, Any],
) -> None:
    """Test HuggingFace OpenResponses with multimodal input (image)."""
    try:
        llm = AnyLLM.create(LLMProvider.HUGGINGFACE, **huggingface_client_config)

        # Use a vision-capable model
        result = await llm.aresponses(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            input_data="What colors do you see in this image? https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/320px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        )

        assert isinstance(result, Response)
        assert result.output_text is not None

    except MissingApiKeyError:
        pytest.skip("HF_TOKEN not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError) as e:
        pytest.skip(f"Network error connecting to HuggingFace router: {e}")
