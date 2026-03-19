# Copilot Instructions for any-llm

> Full guidelines are in [AGENTS.md](../AGENTS.md). This file surfaces the rules most critical for AI-assisted coding.

## Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync --all-extras -U

# Run all checks (preferred before committing)
uv run pre-commit run --all-files --verbose

# Tests
uv run pytest -v tests/unit
uv run pytest -v tests/integration -n auto   # requires API keys
```

## Code Style (enforced by mypy + ruff)

- **Type hints required** on all new code; mypy runs in strict mode
- **`@override` decorator** from `typing_extensions` is required on every method that overrides a base class method — mypy enforces `explicit-override`. For static methods: `@staticmethod` first, then `@override`
- **Direct attribute access** (`obj.field`) preferred over `getattr(obj, "field")` for typed fields
- Line length: 120 chars (ruff)
- No decorative section-separator comments (`# ------` banners)

## Project Structure

```
src/any_llm/
  providers/<provider>/   ← all provider-specific code goes here
  types/                  ← shared types
  gateway/                ← optional FastAPI gateway
tests/
  unit/                   ← no API keys needed
  integration/            ← skip when creds unavailable
  gateway/
```

## Testing Rules

- **No class-based test grouping** — all tests are standalone functions
- Add happy path + error/raise path tests for every change (~85% coverage target)
- Integration tests must `pytest.skip(...)` when credentials are unavailable
- Optional-dependency imports (e.g. `mistralai`, `cohere`) go **inside** the test function, not at the top of the file

## Commits & PRs

- Conventional Commits: `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`
- PRs must complete the checklist in `.github/pull_request_template.md` and include AI-usage disclosure when applicable
- Never commit secrets — use env vars or a gitignored `.env`
