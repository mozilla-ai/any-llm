# ruff: noqa: T201, S104
import asyncio
import json
from typing import Any

from any_llm import acompletion, list_models
from any_llm.exceptions import MissingApiKeyError
from any_llm.provider import ProviderFactory, ProviderName
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="any-llm Demo", description="Demo showcasing list_models and completions")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ListModelsRequest(BaseModel):
    provider: str


class CompletionRequest(BaseModel):
    provider: str
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None


@app.get("/")
async def root():
    return {"message": "any-llm Demo API"}


@app.get("/providers")
async def get_providers():
    """Get all providers that support list_models."""
    supported_providers = []

    for provider_name in ProviderName:
        provider_class = ProviderFactory.get_provider_class(provider_name)
        if provider_class.SUPPORTS_LIST_MODELS:
            supported_providers.append(
                {"name": provider_name.value, "display_name": provider_name.value.replace("_", " ").title()}
            )

    return {"providers": supported_providers}


@app.post("/list-models")
async def get_models(request: ListModelsRequest):
    """List available models for a provider."""
    try:
        models = list_models(provider=request.provider)

        return {
            "models": [
                {
                    "id": model.id,
                    "object": model.object,
                    "created": getattr(model, "created", None),
                    "owned_by": getattr(model, "owned_by", None),
                }
                for model in models
            ]
        }
    except MissingApiKeyError as e:
        raise HTTPException(
            status_code=400,
            detail="API key is required for this provider, please set the env var and restart the backend server",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def stream_completion(request: CompletionRequest):
    """Stream completion chunks as Server-Sent Events."""
    try:
        stream = await acompletion(
            model=request.model,
            messages=request.messages,
            provider=request.provider,
            temperature=request.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                delta_data = {}

                # Handle content
                if choice.delta and hasattr(choice.delta, "content") and choice.delta.content:
                    delta_data["content"] = choice.delta.content

                # Handle thinking/reasoning
                if choice.delta and hasattr(choice.delta, "reasoning") and choice.delta.reasoning:
                    delta_data["thinking"] = choice.delta.reasoning.content

                # Send chunk if we have any delta data
                if delta_data:
                    chunk_data = {
                        "id": chunk.id,
                        "object": chunk.object,
                        "created": chunk.created,
                        "model": chunk.model,
                        "choices": [
                            {"index": choice.index, "delta": delta_data, "finish_reason": choice.finish_reason}
                        ],
                    }
                    chunk_json = json.dumps(chunk_data)
                    yield f"data: {chunk_json}\n\n"

                    # Add a small delay to help with browser rendering
                    await asyncio.sleep(0.01)
                elif choice.finish_reason:
                    final_data = {
                        "id": chunk.id,
                        "object": chunk.object,
                        "created": chunk.created,
                        "model": chunk.model,
                        "choices": [{"index": choice.index, "delta": {}, "finish_reason": choice.finish_reason}],
                    }
                    yield f"data: {json.dumps(final_data)}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@app.post("/completion")
async def create_completion(request: CompletionRequest):
    """Create a streaming completion using the specified model and provider."""
    try:
        return StreamingResponse(
            stream_completion(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )
    except MissingApiKeyError as e:
        raise HTTPException(
            status_code=400,
            detail="API key is required for this provider, please set the env var and restart the backend server",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    print("Starting any-llm demo server...")
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("Stop with Ctrl+C")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
