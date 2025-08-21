# any-llm Demo Backend

FastAPI backend for the any-llm demo showcasing the `list_models` feature.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the development server
uv run python main.py
```

The API will be available at:
- **Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Dependencies

This project uses `uv` for dependency management. All dependencies are defined in `pyproject.toml`:

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **any-llm**: The main library for LLM interactions
- **python-multipart**: For handling form data

## API Endpoints

- `GET /` - Health check
- `GET /providers` - List all providers supporting list_models
- `POST /list-models` - Get available models for a provider
- `POST /completion` - Create a completion using selected model

## Development

The project includes Ruff configuration for linting and formatting. Run linting with:

```bash
uv run ruff check .
uv run ruff format .
```
