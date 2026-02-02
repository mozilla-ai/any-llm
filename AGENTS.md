# any-llm Repository Guidelines

> Python 3.11+ | `uv` (not pip) | `ruff` formatting | `pytest` (no test classes)

## Quick Commands
- **Setup**: `uv venv && source .venv/bin/activate && uv sync --all-extras -U`
- **All checks**: `uv run pre-commit run --all-files --verbose`
- **Unit tests**: `uv run pytest -v tests/unit`
- **Integration tests**: `uv run pytest -v tests/integration -n auto`
- **Specific test**: `uv run pytest -v tests/unit/path/to/test.py::test_name`
- **Docs preview**: `uv run mkdocs serve`
- **Run gateway**: (from `docker/`) `cp config.example.yml config.yml && docker compose up --build`

## Where to Look First

- [README.md](README.md): high-level usage and gateway overview.
- [CONTRIBUTING.md](CONTRIBUTING.md): canonical dev setup, test matrix, and contribution workflow.
- [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml): formatting/lint/typecheck configuration.
- [docs/](docs/): MkDocs site sources (configured by [mkdocs.yml](mkdocs.yml)).

## Project Structure 
```
src/any_llm/    #Core SDK 
├── providers/         # Provider implementations (isolated)
├── types/             # Shared type definitions
└── gateway/           # Optional FastAPI gateway (proxy + budgeting/keys/analytics)
tests/                 # unit/, integration/, gateway/
├──conftest.py         # Shared fixtures
docs/                  # MkDocs documentation site (config in `mkdocs.yml`).
docker/                # Gateway Dockerfile + Compose configs 
demos/                 # Example apps(Python: `demos/*/backend`, React: `demos/*/frontend`).
```

## Coding Style & Naming Conventions

- **Python indentation**: 4 spaces; 
    - Formatting:  `ruff` (line length 120), enforced via `pre-commit`.
- **Types**: Required; `mypy` strict mode for library code (see `pyproject.toml`).
- **Provider-Specific Behavior**: K eep provider-specific behavior code under `src/any_llm/providers/<provider>/`.
- **Access** : Prefer direct attribute access (e.g., `obj.field`) over `getattr(obj, "field")` when the field is typed. Only use `getattr`/`setattr` when working with truly dynamic attributes or when type information is unavailable.
- Comments: Add only if helpful, remove if obvious.
- Simplify: Consolidate or remove unneeded code when possible.

## Testing

- Framework: `pytest` (+ `pytest-asyncio`, `pytest-xdist`).
- Add/adjust tests with every change (happy path + error cases).
- Skip integration tests when credentials/services aren’t available : `pytest.skip(...)`
- Target ~85%+ coverage (see `CONTRIBUTING.md`).
- **No test classes**: Do not use class-based test grouping (`class TestFoo:`). All tests should be standalone functions.
```python
# ✅ Good
def test_provider_returns_response():
    ...

# ❌ Bad
class TestProvider:
    def test_returns_response(self):
        ...
```

## Verification Checklist
Before marking work complete:
- [ ] `uv run pre-commit run --all-files --verbose` passes
- [ ] `uv run pytest -v tests/unit` passes  
- [ ] New code has type hints
- [ ] Tests added (happy path + errors)
- [ ] No debug code or commented-out code left

## Commit & Pull Request Guidelines

- Commits follow the project’s history: Conventional Commits such as `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`.
- PRs should follow [.github/pull_request_template.md](.github/pull_request_template.md): clear description, linked issues (e.g., `Fixes #123`), completed checklist, and AI-usage disclosure when applicable.

## Security & Configuration Tips

- Never commit secrets. Use environment variables or a local `.env` (gitignored) for provider API keys.
