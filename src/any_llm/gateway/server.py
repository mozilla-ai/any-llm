from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from typing_extensions import override

from any_llm.gateway import __version__
from any_llm.gateway.auth.dependencies import set_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import get_db, init_db
from any_llm.gateway.pricing_init import initialize_pricing_from_config
from any_llm.gateway.rate_limit import RateLimiter
from any_llm.gateway.routes import budgets, chat, embeddings, health, keys, messages, models, pricing, users

_PUBLIC_PREFIXES = ("/health",)


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Prevent CDN/proxy caches from storing authenticated responses."""

    @override
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        if not request.url.path.startswith(_PUBLIC_PREFIXES):
            response.headers["Cache-Control"] = "private, no-store, no-cache"
            response.headers["Vary"] = "Authorization"
        return response


def create_app(config: GatewayConfig) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        config: Gateway configuration

    Returns:
        Configured FastAPI application

    """
    init_db(config.database_url, auto_migrate=config.auto_migrate)
    set_config(config)

    db = next(get_db())
    try:
        initialize_pricing_from_config(config, db)
    finally:
        db.close()

    app = FastAPI(
        title="any-llm-gateway",
        description="A clean FastAPI gateway for any-llm with API key management",
        version=__version__,
    )

    app.add_middleware(CacheControlMiddleware)

    if config.cors_allow_origins:
        allow_credentials = "*" not in config.cors_allow_origins
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_allow_origins,
            allow_credentials=allow_credentials,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-AnyLLM-Key", "x-api-key"],
        )

    if config.enable_metrics:
        from any_llm.gateway.metrics import MetricsMiddleware

        app.add_middleware(MetricsMiddleware)

    if config.rate_limit_rpm is not None:
        app.state.rate_limiter = RateLimiter(config.rate_limit_rpm)
    else:
        app.state.rate_limiter = None

    app.include_router(chat.router)
    app.include_router(messages.router)
    app.include_router(embeddings.router)
    app.include_router(models.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(budgets.router)
    app.include_router(pricing.router)
    app.include_router(health.router)

    if config.enable_metrics:
        from any_llm.gateway.metrics import metrics_endpoint

        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app
