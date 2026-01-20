uv sync --group lint
uv run ruff format .
uv run ruff check . --fix
