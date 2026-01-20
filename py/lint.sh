uv sync --group lint
uv run ruff format src
uv run ruff check src --fix
