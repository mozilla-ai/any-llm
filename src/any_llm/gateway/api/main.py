from fastapi import FastAPI

from any_llm.gateway.api.routes import budgets, chat, embeddings, health, keys, messages, models, pricing, users


def register_routers(app: FastAPI) -> None:
    app.include_router(chat.router)
    app.include_router(messages.router)
    app.include_router(embeddings.router)
    app.include_router(models.router)
    app.include_router(keys.router)
    app.include_router(users.router)
    app.include_router(budgets.router)
    app.include_router(pricing.router)
    app.include_router(health.router)
