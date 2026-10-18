from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.AZUREOPENAI])]


@pytest_asyncio.fixture
async def azure_v1(
    provider: LLMProvider,
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> AsyncIterator[AzureopenaiProvider]:
    try:
        llm = AzureopenaiProvider(**provider_client_config[provider])
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Azure v1 credentials missing: set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN")
    try:
        yield llm
    finally:
        await llm.client.close()


async def test_azure_v1_core(
    azure_v1: AzureopenaiProvider,
    provider_model_map: dict[LLMProvider, str],
) -> None:
    model = provider_model_map[LLMProvider.AZUREOPENAI]
    assert azure_v1.client.base_url.path.endswith("/openai/v1/")
    result = await azure_v1.acompletion(model=model, messages=[{"role": "user", "content": "Say hello."}])
    assert result.choices[0].message.content
    response = await azure_v1.aresponses(model=model, input_data="Say hello.")
    assert response.output
    assert await azure_v1.alist_models()
