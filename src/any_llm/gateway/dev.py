from fastapi import FastAPI

from any_llm.gateway.core.config import load_config
from any_llm.gateway.main import create_app


def create_dev_app() -> FastAPI:
    return create_app(load_config())
