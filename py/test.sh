K_FILTER="${1:-not integration}"

uv sync --group tests --extra all
uv run pytest -vv tests -k "$K_FILTER"
