import asyncio
import os
import time
from typing import Any

import httpx
import pytest
import pytest_asyncio

from any_llm import AnyLLM, Providers

BASE_URL = os.getenv("TEST_SERVER_URL", "http://localhost:8080/v1")
DUMMY_API_KEY = "test-key"
PROVIDERS_TO_TEST = [Providers.OPENAI]


@pytest_asyncio.fixture(scope="session")
async def test_run_id() -> str:
    return f"py-{int(time.time() * 1000)}"


@pytest_asyncio.fixture(scope="session")
async def scenarios() -> dict[str, Any]:
    server_base = BASE_URL.replace("/v1", "")
    url = f"{server_base}/v1/test-data"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            pytest.fail(f"Failed to load test scenarios: {response.status_code}")
        data = response.json()
        return data["scenarios"]


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_run(test_run_id: str) -> None:
    server_base = BASE_URL.replace("/v1", "")
    url = f"{server_base}/v1/test-runs?test_run_id={test_run_id}&description=Python%20acceptance%20tests"

    async with httpx.AsyncClient() as client:
        response = await client.post(url)
        if response.status_code not in (200, 409):
            pytest.skip(f"Could not create test run: {response.status_code}")


def pytest_generate_tests(metafunc):
    if "scenario_name" in metafunc.fixturenames:

        async def _load():
            server_base = BASE_URL.replace("/v1", "")
            url = f"{server_base}/v1/test-data"
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    return list(data["scenarios"].keys())
            return []

        scenario_names = asyncio.run(_load())
        if scenario_names:
            metafunc.parametrize("scenario_name", scenario_names)

    if "provider" in metafunc.fixturenames:
        metafunc.parametrize("provider", PROVIDERS_TO_TEST)


@pytest.mark.asyncio
async def test_scenario(
    provider: str,
    test_run_id: str,
    scenarios: dict[str, Any],
    scenario_name: str,
) -> None:
    llm_client = await AnyLLM.acreate(
        provider,
        DUMMY_API_KEY,
        BASE_URL,
        default_headers={"X-Test-Run-Id": test_run_id},
    )

    scenario = scenarios[scenario_name]

    if scenario.get("stream", False):
        chunks = []
        options = scenario.get("options", {})

        async for chunk in llm_client.acompletion_stream(
            scenario["model"],
            scenario["messages"],
            **options,
        ):
            chunks.append(chunk)

        assert chunks, "No chunks received in streaming response"
    else:
        options = scenario.get("options", {})
        response = await llm_client.acompletion(
            scenario["model"],
            scenario["messages"],
            **options,
        )

        assert response.get("choices"), "Response missing choices"
        assert len(response["choices"]) > 0, "Response has no choices"
