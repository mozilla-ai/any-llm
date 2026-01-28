# Repository Guidelines

## Where to Look First

- [README.md](README.md): high-level usage and gateway overview.
- [CONTRIBUTING.md](CONTRIBUTING.md): canonical dev setup, test matrix, and contribution workflow.
- [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml): formatting/lint/typecheck configuration.
- [docs/](docs/): MkDocs site sources (configured by [mkdocs.yml](mkdocs.yml)).

## Project Structure & Module Organization

- `src/any_llm/`: Python SDK source (providers in `src/any_llm/providers/`, shared types in `src/any_llm/types/`).
- `src/any_llm/gateway/`: Optional FastAPI gateway (OpenAI-compatible proxy + budgeting/keys/analytics).
- `tests/`: `unit/`, `integration/`, `gateway/`, plus shared fixtures in `tests/conftest.py`.
- `docs/`: MkDocs documentation site (config in `mkdocs.yml`).
- `docker/`: Gateway Dockerfile + Compose configs ([docker/docker-compose.yml](docker/docker-compose.yml), [docker/config.example.yml](docker/config.example.yml)).
- `demos/`: Example apps (`demos/*/backend` in Python, `demos/*/frontend` in React).

## Build, Test, and Development Commands

This repo uses `uv` for local dev (Python 3.11+). For the full, up-to-date command set, follow [CONTRIBUTING.md](CONTRIBUTING.md).

- Create env + install dev deps: `uv venv && source .venv/bin/activate && uv sync --all-extras -U`
- Run all checks (preferred): `uv run pre-commit run --all-files --verbose`
- Unit tests: `uv run pytest -v tests/unit`
- Integration tests (often require API keys): `uv run pytest -v tests/integration -n auto`
- Docs preview: `uv run mkdocs serve`
- Run gateway via Docker (from `docker/`): `cp config.example.yml config.yml && docker compose up --build`

## Coding Style & Naming Conventions

- Python indentation: 4 spaces; formatting/linting via `ruff` (line length 120) and `pre-commit`.
- Type hints: required; `mypy` runs in strict mode for library code (see `pyproject.toml`).
- Provider code lives under `src/any_llm/providers/<provider>/` (keep provider-specific behavior isolated there).
- **Override decorator**: When overriding methods from base classes (like `AnyLLM`), always use the `@override` decorator from `typing_extensions`. This is enforced by mypy's `explicit-override` error code. For static methods, the order is `@staticmethod` followed by `@override`.
- Prefer direct attribute access (e.g., `obj.field`) over `getattr(obj, "field")` when the field is typed. This enables `ruff` and `mypy` to catch errors at lint time. Only use `getattr`/`setattr` when working with truly dynamic attributes or when type information is unavailable.
- Please add code comments if you find them helpful to accomplish your objective. However, please remove any comments you added that describe obvious behavior before finishing your task.

## Testing Guidelines

- Framework: `pytest` (+ `pytest-asyncio`, `pytest-xdist`).
- Add/adjust tests with every change (happy path + error cases). Integration tests should `pytest.skip(...)` when credentials/services aren’t available.
- New code should target ~85%+ coverage (see `CONTRIBUTING.md`).
- Do not use class-based test grouping (`class TestFoo:`). All tests should be standalone functions.

## Commit & Pull Request Guidelines

- Commits follow the project’s history: Conventional Commits such as `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`.
- PRs should follow [.github/pull_request_template.md](.github/pull_request_template.md): clear description, linked issues (e.g., `Fixes #123`), completed checklist, and AI-usage disclosure when applicable.

## Security & Configuration Tips

- Never commit secrets. Use environment variables or a local `.env` (gitignored) for provider API keys.
