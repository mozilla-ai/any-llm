import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db.models import UsageLog, User
from tests.gateway.conftest import MODEL_NAME


@pytest.mark.asyncio
async def test_completion_accuracy(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_config: GatewayConfig,
    test_user: dict[str, Any],
    test_messages: list[dict[str, str]],
    model_pricing: dict[str, Any],
) -> None:
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Capture initial user spend
    db = session_local()
    try:
        user = db.query(User).filter(User.user_id == test_user["user_id"]).first()
        initial_spend = float(user.spend) if user else 0.0
    finally:
        db.close()

    # Make completion request
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages,
            "user": test_user["user_id"],
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    response_data = response.json()

    # Extract usage from response
    assert "usage" in response_data, "Response should contain usage data"
    usage = response_data["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]
    total_tokens = usage[
        "total_tokens"
    ]  # total isn't = prompt_tokens + completion_tokens, it's prompt + completion + reasoning

    # Calculate expected cost
    input_price = model_pricing["input_price_per_million"]
    output_price = model_pricing["output_price_per_million"]
    expected_cost = (prompt_tokens / 1_000_000) * input_price + (completion_tokens / 1_000_000) * output_price

    db = session_local()
    try:
        # Check usage log
        usage_logs = db.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).all()
        assert len(usage_logs) > 0
        log = usage_logs[0]

        assert log.model == "gemini-2.5-flash"
        assert log.provider == "gemini"
        assert log.endpoint == "/v1/chat/completions"
        assert log.status == "success"
        assert log.total_tokens is not None
        assert log.total_tokens > 0

        # Verify token counts match
        assert log.prompt_tokens == prompt_tokens, (
            f"Logged prompt tokens {log.prompt_tokens} != response {prompt_tokens}"
        )
        assert log.completion_tokens == completion_tokens, (
            f"Logged completion tokens {log.completion_tokens} != response {completion_tokens}"
        )
        assert log.total_tokens == total_tokens, f"Logged total tokens {log.total_tokens} != response {total_tokens}"

        # Verify cost calculation
        assert log.cost is not None, "Cost should be logged when pricing is configured"
        assert abs(log.cost - expected_cost) < 0.0001, (
            f"Logged cost {log.cost} does not match expected cost {expected_cost}. "
            f"Calculation: ({prompt_tokens}/1M * ${input_price}) + ({completion_tokens}/1M * ${output_price})"
        )

        # Check user spend
        user = db.query(User).filter(User.user_id == test_user["user_id"]).first()
        assert user is not None
        final_spend = float(user.spend)

        assert final_spend > initial_spend, "User spend should increase after request"
        cost_increase = final_spend - initial_spend

        # Verify the exact cost increase matches our calculation
        assert abs(cost_increase - expected_cost) < 0.0001, (
            f"User spend increase {cost_increase} does not match expected cost {expected_cost}. "
            f"Calculation: ({prompt_tokens}/1M * ${input_price}) + ({completion_tokens}/1M * ${output_price})"
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_streaming_completion_accuracy(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_config: GatewayConfig,
    test_user: dict[str, Any],
    test_messages_with_longer_response: list[dict[str, str]],
    model_pricing: dict[str, Any],
) -> None:
    """Test that streaming requests correctly aggregate usage, calculate costs, and update user spend."""
    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Capture initial user spend
    db = session_local()
    try:
        user = db.query(User).filter(User.user_id == test_user["user_id"]).first()
        initial_spend = float(user.spend) if user else 0.0
    finally:
        db.close()

    # Make streaming request
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": test_messages_with_longer_response,
            "user": test_user["user_id"],
            "stream": True,
        },
        headers=api_key_header,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Aggregate usage data from all chunks (matching server behavior)
    aggregated_prompt_tokens = 0
    aggregated_completion_tokens = 0
    aggregated_total_tokens = 0
    chunks_with_usage = 0

    for line in response.iter_lines():
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break

            try:
                chunk = json.loads(data_str)
                if "usage" in chunk and chunk["usage"] is not None:
                    usage = chunk["usage"]
                    # Match server aggregation logic:
                    # - prompt_tokens: take first non-zero value
                    # - completion_tokens: take maximum (cumulative)
                    # - total_tokens: take maximum (cumulative)
                    if "prompt_tokens" in usage and usage["prompt_tokens"] and not aggregated_prompt_tokens:
                        aggregated_prompt_tokens = usage["prompt_tokens"]
                    if usage.get("completion_tokens"):
                        aggregated_completion_tokens = max(aggregated_completion_tokens, usage["completion_tokens"])
                    if usage.get("total_tokens"):
                        aggregated_total_tokens = max(aggregated_total_tokens, usage["total_tokens"])
                    chunks_with_usage += 1
            except json.JSONDecodeError:
                continue

    assert chunks_with_usage > 0, "Should have received at least one chunk with usage data"

    # Calculate expected cost from aggregated tokens
    input_price = model_pricing["input_price_per_million"]
    output_price = model_pricing["output_price_per_million"]
    expected_cost = (aggregated_prompt_tokens / 1_000_000) * input_price + (
        aggregated_completion_tokens / 1_000_000
    ) * output_price

    await asyncio.sleep(1)

    # Verify usage log and user spend
    db = session_local()
    try:
        # Check usage log
        usage_logs = db.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).all()
        assert len(usage_logs) > 0, "No usage logs found for streaming request"
        log = usage_logs[0]

        assert log.model == "gemini-2.5-flash"
        assert log.provider == "gemini"
        assert log.endpoint == "/v1/chat/completions"
        assert log.status == "success"
        assert log.total_tokens is not None, "Total tokens should be logged for streaming requests"
        assert log.total_tokens > 0, "Total tokens should be greater than 0"
        assert log.prompt_tokens is not None, "Prompt tokens should be logged"
        assert log.completion_tokens is not None, "Completion tokens should be logged"
        assert log.prompt_tokens > 0, "Prompt tokens should be greater than 0"
        assert log.completion_tokens > 0, "Completion tokens should be greater than 0"

        # Verify aggregated tokens match what was logged
        assert log.prompt_tokens == aggregated_prompt_tokens, (
            f"Logged prompt tokens {log.prompt_tokens} != aggregated from chunks {aggregated_prompt_tokens}"
        )
        assert log.completion_tokens == aggregated_completion_tokens, (
            f"Logged completion tokens {log.completion_tokens} != aggregated from chunks {aggregated_completion_tokens}"
        )
        assert log.total_tokens == aggregated_total_tokens, (
            f"Logged total tokens {log.total_tokens} != aggregated from chunks {aggregated_total_tokens}"
        )

        # Verify cost calculation based on aggregated tokens
        assert log.cost is not None, "Cost should be logged when pricing is configured"
        assert abs(log.cost - expected_cost) < 0.0001, (
            f"Logged cost {log.cost} does not match expected cost {expected_cost} from aggregated tokens. "
            f"Calculation: ({aggregated_prompt_tokens}/1M * ${input_price}) + ({aggregated_completion_tokens}/1M * ${output_price})"
        )

        # Check user spend
        user = db.query(User).filter(User.user_id == test_user["user_id"]).first()
        assert user is not None
        final_spend = float(user.spend)

        assert final_spend > initial_spend, "User spend should increase after streaming request"
        cost_increase = final_spend - initial_spend

        # Verify the exact cost increase matches our calculation
        assert abs(cost_increase - expected_cost) < 0.0001, (
            f"User spend increase {cost_increase} does not match expected cost {expected_cost}. "
            f"Calculation: ({aggregated_prompt_tokens}/1M * ${input_price}) + ({aggregated_completion_tokens}/1M * ${output_price})"
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_failed_request_logs_error(
    client: TestClient,
    api_key_header: dict[str, str],
    api_key_obj: dict[str, Any],
    test_config: GatewayConfig,
    test_user: dict[str, Any],
    test_messages: list[dict[str, str]],
) -> None:
    """Test that failed requests are logged with error status."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini:invalid-model",
            "messages": test_messages,
            "user": test_user["user_id"],
        },
        headers=api_key_header,
    )

    assert response.status_code == 500

    engine = create_engine(test_config.database_url)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = session_local()

    try:
        usage_logs = db.query(UsageLog).filter(UsageLog.api_key_id == api_key_obj["id"]).all()

        assert len(usage_logs) > 0
        log = usage_logs[0]

        assert log.status == "error"
        assert log.error_message is not None
    finally:
        db.close()
