import asyncio
import base64
from pathlib import Path
from typing import Any

import aiofiles
import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_async_completion(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")

        model_id = provider_model_map[provider]
        result = await llm.acompletion(
            model=model_id,
            messages=[
                {"role": "user", "content": "Hello"},
                ChatCompletionMessage(role="assistant", content="Hi!"),
                {"role": "user", "content": "What is my name?"},
            ],
        )
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local Model host is not set up, skipping")
        raise
    assert isinstance(result, ChatCompletion)
    assert result.choices[0].message.content is not None
    assert hasattr(
        result.choices[0].message, "reasoning"
    )  # If all the providers are properly implementing the reasoning, this should be true


@pytest.mark.asyncio
async def test_async_completion_parallel(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION:
            pytest.skip(f"{provider.value} does not support completion, skipping")

        model_id = provider_model_map[provider]
        prompt_1 = "What's the capital of France?"
        prompt_2 = "What's the capital of Germany?"
        results = await asyncio.gather(
            llm.acompletion(model=model_id, messages=[{"role": "user", "content": prompt_1}]),
            llm.acompletion(model=model_id, messages=[{"role": "user", "content": prompt_2}]),
        )
        assert isinstance(results[0], ChatCompletion)
        assert isinstance(results[1], ChatCompletion)
        assert results[0].choices[0].message.content is not None
        assert results[1].choices[0].message.content is not None
        assert "paris" in results[0].choices[0].message.content.lower()
        assert "berlin" in results[1].choices[0].message.content.lower()
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_completion_with_image(
    provider: LLMProvider,
    provider_image_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    try:
        if provider == LLMProvider.LLAMACPP:
            pytest.skip("We use a llamacpp model that doesn't support images, skipping")
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_IMAGE:
            pytest.skip(f"{provider.value} does not support completion, skipping")

        model_id = provider_image_model_map[provider]
        assets_dir = Path(__file__).parent.parent / "assets"
        async with aiofiles.open(assets_dir / "any-llm-logo.png", "rb") as image_file:
            image_data = await image_file.read()
            base64_img = base64.b64encode(image_data).decode("utf-8")
        response = await llm.acompletion(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at the image: does it say any-llm?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        },
                    ],
                }
            ],
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_completion_with_pdf(
    provider: LLMProvider,
    provider_image_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_COMPLETION_PDF:
            pytest.skip(f"{provider.value} does not support completion, skipping")

        model_id = provider_image_model_map[provider]
        assets_dir = Path(__file__).parent.parent / "assets"
        async with aiofiles.open(assets_dir / "cv_1.pdf", "rb") as pdf_file:
            pdf_data = await pdf_file.read()
            base64_pdf = base64.b64encode(pdf_data).decode("utf-8")
        data_url = f"data:application/pdf;base64,{base64_pdf}"
        response = await llm.acompletion(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at the pdf: does it say any-llm?"},
                        {
                            "type": "file",
                            "file": {
                                "filename": "document.pdf",
                                "file_data": data_url,
                            },
                        },
                    ],
                }
            ],
        )
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise
