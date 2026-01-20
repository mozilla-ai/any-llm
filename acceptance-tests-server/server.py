# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.115.0",
#     "pydantic>=2.0.0",
#     "uvicorn>=0.30.0",
# ]
# ///
"""FastAPI acceptance test server for any-llm client validation."""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from database import db
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResultsResponse,
    ScenarioID,
    TestRun,
    TestRunSummary,
)
from responses import create_mock_response, create_streaming_response
from scenarios import get_all_scenarios, get_scenario_for_model
from validators import validate_request


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    """Application lifespan handler."""
    yield


app = FastAPI(
    title="any-llm Acceptance Test Server",
    description="Mock LLM provider for validating any-llm client implementations",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/scenarios")
async def list_scenarios() -> list[dict[str, Any]]:
    """List all available test scenarios."""
    return [s.model_dump() for s in get_all_scenarios()]


@app.post("/v1/test-runs")
async def create_test_run(
    test_run_id: str | None = Query(default=None, description="Custom test run ID (auto-generated if not provided)"),
    description: str | None = Query(default=None, description="Description of the test run"),
) -> TestRun:
    """Create a new test run to group validation results."""
    run_id = test_run_id or f"run-{uuid.uuid4().hex[:12]}"

    existing = db.get_test_run(run_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Test run '{run_id}' already exists")

    result = db.create_test_run(run_id, description)
    return TestRun(**result)


@app.get("/v1/test-runs")
async def list_test_runs(
    limit: int = Query(default=100, description="Maximum number of test runs to return"),
) -> list[TestRun]:
    """List all test runs."""
    runs = db.list_test_runs(limit)
    return [TestRun(**r) for r in runs]


@app.get("/v1/test-runs/{test_run_id}")
async def get_test_run(test_run_id: str) -> TestRun:
    """Get a specific test run."""
    run = db.get_test_run(test_run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Test run '{test_run_id}' not found")
    return TestRun(**run)


@app.delete("/v1/test-runs/{test_run_id}")
async def delete_test_run(test_run_id: str) -> dict[str, str]:
    """Delete a test run and all its results."""
    deleted = db.delete_test_run(test_run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Test run '{test_run_id}' not found")
    return {"status": "deleted", "test_run_id": test_run_id}


@app.get("/v1/test-runs/{test_run_id}/summary")
async def get_test_run_summary(test_run_id: str) -> TestRunSummary:
    """Get a summary of results for a specific test run."""
    run = db.get_test_run(test_run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Test run '{test_run_id}' not found")

    summary = db.get_summary(test_run_id)
    return TestRunSummary(test_run_id=test_run_id, **summary)


@app.get("/v1/test-runs/{test_run_id}/results")
async def get_test_run_results(
    test_run_id: str,
    scenario: ScenarioID | None = Query(default=None, description="Filter by scenario"),
    passed: bool | None = Query(default=None, description="Filter by pass/fail status"),
    limit: int = Query(default=1000, description="Maximum number of results to return"),
) -> ResultsResponse:
    """Get validation results for a specific test run."""
    run = db.get_test_run(test_run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Test run '{test_run_id}' not found")

    results = db.get_results(test_run_id=test_run_id, scenario=scenario, passed=passed, limit=limit)

    return ResultsResponse(
        test_run_id=test_run_id,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        results=results,
    )


@app.get("/v1/results")
async def get_results(
    test_run_id: str | None = Query(default=None, description="Filter by test run ID"),
    scenario: ScenarioID | None = Query(default=None, description="Filter by scenario"),
    passed: bool | None = Query(default=None, description="Filter by pass/fail status"),
    limit: int = Query(default=1000, description="Maximum number of results to return"),
) -> ResultsResponse:
    """Get validation results with optional filters."""
    results = db.get_results(test_run_id=test_run_id, scenario=scenario, passed=passed, limit=limit)

    return ResultsResponse(
        test_run_id=test_run_id,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        results=results,
    )


@app.get("/v1/summary")
async def get_summary(
    test_run_id: str | None = Query(default=None, description="Filter by test run ID"),
) -> TestRunSummary:
    """Get a summary of all results, optionally filtered by test run."""
    summary = db.get_summary(test_run_id)
    return TestRunSummary(test_run_id=test_run_id, **summary)


@app.delete("/v1/results")
async def clear_results() -> dict[str, str]:
    """Clear all validation results and test runs."""
    db.clear_all()
    return {"status": "cleared"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request,
    x_test_run_id: str | None = Header(default=None, description="Test run ID to associate this request with"),
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completions endpoint with validation.

    Pass X-Test-Run-Id header to associate this request with a test run.
    If the test run doesn't exist, it will be created automatically.
    """
    body = await request.json()

    test_run_id = x_test_run_id or f"default-{uuid.uuid4().hex[:8]}"

    try:
        completion_request = ChatCompletionRequest.model_validate(body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}") from e

    scenario = get_scenario_for_model(completion_request.model)
    if scenario is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{completion_request.model}'. Use a test-* model name to select a scenario.",
        )

    request_id = f"req-{uuid.uuid4().hex[:8]}"

    validation = validate_request(scenario, completion_request, request_id)
    validation.timestamp = time.time()
    validation.test_run_id = test_run_id

    db.store_result(test_run_id, validation, body)

    if completion_request.stream:

        async def stream_generator():  # noqa: ANN202
            async for chunk in create_streaming_response(completion_request, scenario, validation):
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Validation-Passed": str(validation.passed).lower(),
                "X-Scenario": scenario.value,
                "X-Test-Run-Id": test_run_id,
                "X-Request-Id": request_id,
            },
        )

    return create_mock_response(completion_request, scenario, validation)


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """List available test models."""
    from models import MODEL_TO_SCENARIO

    models = []
    for model_name, scenario in MODEL_TO_SCENARIO.items():
        models.append(
            {
                "id": model_name,
                "object": "model",
                "created": 1700000000,
                "owned_by": "acceptance-test-server",
                "description": f"Test model for {scenario.value} scenario",
            }
        )

    return {"object": "list", "data": models}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
