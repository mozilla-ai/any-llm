# ruff: noqa: T201, S104
import os

from any_llm import AnyLLM, LLMProvider, alist_models
from any_llm.exceptions import MissingApiKeyError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="any-llm Model Finder", description="Find models across different providers")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str


class ProviderStatus(BaseModel):
    name: str
    display_name: str
    api_key_configured: bool
    env_var: str
    supports_list_models: bool
    missing_packages: bool
    error: str | None = None


class ModelInfo(BaseModel):
    id: str
    provider: str
    provider_display_name: str
    object: str | None = None
    created: int | None = None
    owned_by: str | None = None


@app.get("/")
async def root():
    return {"message": "any-llm Model Finder API"}


@app.get("/provider-status")
async def get_provider_status():
    """Get status of all providers including API key configuration."""
    provider_statuses = []

    for provider_name in LLMProvider:
        try:
            provider_class = AnyLLM.get_provider_class(provider_name)

            status = ProviderStatus(
                name=provider_name.value,
                display_name=provider_name.value.replace("_", " ").title(),
                api_key_configured=bool(os.getenv(provider_class.ENV_API_KEY_NAME)),
                env_var=provider_class.ENV_API_KEY_NAME,
                supports_list_models=provider_class.SUPPORTS_LIST_MODELS,
                missing_packages=provider_class.MISSING_PACKAGES_ERROR is not None,
                error=str(provider_class.MISSING_PACKAGES_ERROR) if provider_class.MISSING_PACKAGES_ERROR else None,
            )

            provider_statuses.append(status)

        except Exception as e:
            status = ProviderStatus(
                name=provider_name.value,
                display_name=provider_name.value.replace("_", " ").title(),
                api_key_configured=False,
                env_var="Unknown",
                supports_list_models=False,
                missing_packages=True,
                error=str(e),
            )
            provider_statuses.append(status)

    return {"providers": provider_statuses}


@app.post("/search-models")
async def search_models(request: SearchRequest):
    """Search for models across all configured providers."""
    query = request.query.lower().strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    all_models = []
    provider_errors = []

    for provider_name in LLMProvider:
        try:
            provider_class = AnyLLM.get_provider_class(provider_name)

            # Skip providers that don't support list_models
            if not provider_class.SUPPORTS_LIST_MODELS:
                continue

            # Skip providers with missing packages
            if provider_class.MISSING_PACKAGES_ERROR is not None:
                continue

            # Skip providers without API keys configured
            if not os.getenv(provider_class.ENV_API_KEY_NAME):
                continue

            try:
                models = await alist_models(provider=provider_name.value)

                for model in models:
                    # Check if the model matches the search query
                    if query in model.id.lower():
                        model_info = ModelInfo(
                            id=model.id,
                            provider=provider_name.value,
                            provider_display_name=provider_name.value.replace("_", " ").title(),
                            object=getattr(model, "object", None),
                            created=getattr(model, "created", None),
                            owned_by=getattr(model, "owned_by", None),
                        )
                        all_models.append(model_info)

            except MissingApiKeyError:
                provider_errors.append({"provider": provider_name.value, "error": "API key not configured"})
            except Exception as e:
                provider_errors.append({"provider": provider_name.value, "error": str(e)})

        except Exception as e:
            provider_errors.append({"provider": provider_name.value, "error": f"Failed to load provider: {e!s}"})

    # Sort models by provider name, then by model name
    all_models.sort(key=lambda x: (x.provider, x.id))

    return {
        "query": request.query,
        "models": all_models,
        "total_found": len(all_models),
        "provider_errors": provider_errors,
    }


@app.get("/all-models")
async def get_all_models():
    """Get all models from all configured providers with streaming updates."""
    import json

    from fastapi.responses import StreamingResponse

    async def stream_all_models():
        """Stream models as each provider completes."""
        all_models = []
        provider_errors = []
        completed_providers = 0

        # First, get list of providers to process
        providers_to_process = []
        for provider_name in LLMProvider:
            try:
                provider_class = AnyLLM.get_provider_class(provider_name)

                # Skip providers that don't support list_models
                if not provider_class.SUPPORTS_LIST_MODELS:
                    continue

                # Skip providers with missing packages
                if provider_class.MISSING_PACKAGES_ERROR is not None:
                    continue

                # Skip providers without API keys configured
                if not os.getenv(provider_class.ENV_API_KEY_NAME):
                    continue

                providers_to_process.append((provider_name, provider_class))
            except Exception:
                print(f"Failed to load provider: {provider_name.value}")
                continue

        total_providers = len(providers_to_process)

        # Send initial status
        yield f"data: {json.dumps({'type': 'status', 'message': f'Loading models from {total_providers} providers...', 'progress': 0, 'total': total_providers})}\n\n"

        # Process each provider
        for provider_name, _ in providers_to_process:
            try:
                provider_display = provider_name.value.replace("_", " ").title()
                yield f"data: {json.dumps({'type': 'status', 'message': f'Loading {provider_display}...', 'progress': completed_providers, 'total': total_providers})}\n\n"

                models = await alist_models(provider=provider_name.value)

                provider_models = []
                for model in models:
                    model_info = ModelInfo(
                        id=model.id,
                        provider=provider_name.value,
                        provider_display_name=provider_display,
                        object=getattr(model, "object", None),
                        created=getattr(model, "created", None),
                        owned_by=getattr(model, "owned_by", None),
                    )
                    all_models.append(model_info)
                    provider_models.append(model_info)

                completed_providers += 1

                # Send provider completion update
                yield f"data: {json.dumps({'type': 'provider_complete', 'provider': provider_name.value, 'provider_display': provider_display, 'models': [model.dict() for model in provider_models], 'progress': completed_providers, 'total': total_providers})}\n\n"

            except MissingApiKeyError:
                error_msg = "API key not configured"
                provider_errors.append({"provider": provider_name.value, "error": error_msg})
                completed_providers += 1
                yield f"data: {json.dumps({'type': 'provider_error', 'provider': provider_name.value, 'provider_display': provider_name.value.replace('_', ' ').title(), 'error': error_msg, 'progress': completed_providers, 'total': total_providers})}\n\n"
            except Exception as e:
                error_msg = str(e)
                provider_errors.append({"provider": provider_name.value, "error": error_msg})
                completed_providers += 1
                yield f"data: {json.dumps({'type': 'provider_error', 'provider': provider_name.value, 'provider_display': provider_name.value.replace('_', ' ').title(), 'error': error_msg, 'progress': completed_providers, 'total': total_providers})}\n\n"

        # Sort models by provider name, then by model name
        all_models.sort(key=lambda x: (x.provider, x.id))

        # Send final result
        yield f"data: {json.dumps({'type': 'complete', 'models': [model.dict() for model in all_models], 'total_models': len(all_models), 'provider_errors': provider_errors})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_all_models(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting any-llm Model Finder server...")
    print("API will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    print("Stop with Ctrl+C")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
