from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from any_llm.gateway import __version__
from any_llm.gateway.auth.dependencies import set_config
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import get_db, init_db
from any_llm.gateway.pricing_init import initialize_pricing_from_config
from any_llm.gateway.routes import budgets, chat, health, keys, pricing, users


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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(budgets.router)
    app.include_router(pricing.router)
    app.include_router(health.router)

    return app
